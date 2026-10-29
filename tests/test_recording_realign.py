"""world_engine_client.recording_realign: a demonstration recording of format
3 or older (obs and states one step late) rewritten in format 4's layout.

The engine is not here, so the tests write their own recordings. _format_4
lays out the steps of a made-up world as the platform's recorder writes
format 4: obs[t] and states[t] before actions[t], next_obs[t] after it.
_as_old makes the old file of the same steps, as the recorder wrote them up
to format 3: obs[t] after actions[t] (format 4's next_obs[t]), states[t] the
state after it (the last one, never in a format 4 file, NaN: realigning
drops it), next_obs[t] = obs[t + 1] with the last repeated. Realigned, each
demo is the format 4 demo without its first row, the one whose observation
the old recorder never wrote, and that row's action side is in
realigned_first/: from the reset, it takes the world to states[0]. A demo of
one sample is left out, and so is its name from the mask lists. A staged
reward's paid_at_start takes in the moved reward. The helper refuses a file
of format 4 or later, one that was not written the old way (realigned
already, say), and to write over a file.

The recordings need h5py, which the package does not install: without it
these tests are skipped, except the two about what the helper does when
h5py does not import.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from world_engine_client import recording_realign
from world_engine_client.recording_realign import realign_recording
from world_engine_client.rotations import axisangle_to_matrix, matrix_to_rot6d

try:
    import h5py
except ImportError:  # the CI test job installs no h5py
    h5py = None

needs_h5py = pytest.mark.skipif(
    h5py is None, reason="h5py is not installed (pip install h5py): the recordings are HDF5"
)

# The made-up world: 8 state values, which each 7-value action moves by a
# fixed linear map, and an observation read from the state.
_MIX = np.random.default_rng(1).normal(size=(8, 7))
# The datasets of a demo that hold the state before the action: the rest of
# the rows belong to the action, and realigned_first/ keeps their first row.
_BEFORE = ("states",)
# The helper's messages for a file it will not rewrite.
_DONE = "already come before each action"
_AGAIN = "it was realigned already"
_NOT_OLD = "is not obs shifted by one step"


class _World:
    """The same seed and actions give the same states, bit for bit."""

    def reset(self, seed: int) -> np.ndarray:
        self.state = np.random.default_rng(seed).normal(size=8)
        return self.state.copy()

    def step(self, action) -> np.ndarray:
        self.state = self.state + _MIX @ np.asarray(action, dtype=np.float64)
        return self.state.copy()


def _obs(state: np.ndarray, objects: bool) -> dict[str, np.ndarray]:
    """robosuite-named, as the recorder names them; ``object`` is empty in a
    scene without movable objects."""
    quat = state[3:7] / np.linalg.norm(state[3:7])
    obj = [state[0] + 0.5, state[1], 0.02, 0.0, 0.0, 0.0, 1.0] if objects else []
    return {
        "robot0_eef_pos": state[:3].astype(np.float32),
        "robot0_eef_quat": quat.astype(np.float32),
        "robot0_gripper_qpos": state[7:8].astype(np.float32),
        "object": np.asarray(obj, dtype=np.float32),
    }


def _steps(seed, rows, *, start_step=0, rewards=None, summary=None, objects=True) -> dict:
    """``rows`` steps of the made-up world, in format 4's rows: from a reset
    with ``seed``, after ``start_step`` steps that are in no row."""
    world, rng = _World(), np.random.default_rng([seed, start_step, rows])
    world.reset(seed)
    for _ in range(start_step):
        world.step(rng.uniform(-1, 1, 7).astype(np.float32))
    d = {"states": [], "obs": [], "next_obs": [], "actions": [], "actions_abs": []}
    for _ in range(rows):
        action = rng.uniform(-1, 1, 7).astype(np.float32)
        d["states"].append(world.state.copy())
        d["obs"].append(_obs(world.state, objects))
        after = world.step(action)
        d["next_obs"].append(_obs(after, objects))
        d["actions"].append(action)
        # The target the action set: [pos, rot_axisangle, gripper].
        d["actions_abs"].append(np.concatenate([after[:3], 0.1 * after[3:6], action[6:]]))
    d["rewards"] = np.zeros(rows) if rewards is None else np.asarray(rewards, dtype=np.float64)
    d["task_progress"] = [{"step": start_step + t + 1} for t in range(rows)]
    d.update(seed=seed, start_step=start_step, summary=summary)
    return d


def _staged(earned, shaping, paid_at_start) -> dict:
    """A staged reward's episode_summary: 4 points in all."""
    reward = {"earned": earned, "max": 4.0, "penalty": 0.0, "shaping": {"return": shaping}}
    return {
        "reward_type": "staged",
        "task_progress": {"reward": reward},
        "paid_at_start": paid_at_start,
    }


def _session() -> dict[str, dict]:
    """A session with a staged task: demo_0 from the reset, its first step
    paying a point; demo_1 one step; demo_2 from step 7 on, after a point was
    paid, with distance shaping."""
    return {
        "demo_0": _steps(
            0, 5, rewards=[0.25, 0.0, 0.125, 0.0, 0.125], summary=_staged(2.0, 0.0, 0.0)
        ),
        "demo_1": _steps(0, 1, rewards=[0.25], summary=_staged(1.0, 0.0, 0.0)),
        "demo_2": _steps(
            0, 3, start_step=7, rewards=[0.25, 0.015625, 0.0], summary=_staged(2.0, 0.0625, 1.0)
        ),
    }


def _rot6d(rotvecs: np.ndarray) -> np.ndarray:
    return np.stack([matrix_to_rot6d(axisangle_to_matrix(v)) for v in rotvecs]).astype(np.float32)


def _format_4(path: Path, demos: dict[str, dict]) -> None:
    """The demos as the platform's recorder writes format 4, without gaps:
    each demo's next_obs[t] is its obs[t + 1]."""
    env_kwargs = {
        "format_version": 4,
        "scene_id": "made_up",
        "physics": "mujoco",
        "control_mode": "osc_pose_delta",
        "action_parts": [["dpos", 3], ["drot", 3], ["gripper", 1]],
        "arm_frame": "world",
        "robot_preset": "franka_panda",
        "control_interval": 0.034,
        "timestep": 0.002,
        "nstep": 17,
        "ee_step": 0.05,
        "ee_rot_step": 0.5,
        "obs_quat_convention": "xyzw",
        "rot_6d_convention": "first_two_rows_gram_schmidt",
    }
    env_args = {"env_name": "WorldEngine_made_up", "type": 1, "env_kwargs": env_kwargs}
    with h5py.File(path, "w") as f:
        data = f.create_group("data")
        data.attrs["total"] = sum(len(d["actions"]) for d in demos.values())
        data.attrs["env_args"] = json.dumps(env_args)
        for name, d in demos.items():
            n = len(d["actions"])
            g = data.create_group(name)
            g.attrs["num_samples"] = n
            g.attrs["control_mode"] = "osc_pose_delta"
            g.attrs["episode_summary"] = json.dumps(d["summary"])
            g.attrs["start_step"] = d["start_step"]
            g.attrs["reset_seed"] = d["seed"]
            g.attrs["gaps"] = np.zeros(0, dtype=np.int64)
            g.create_dataset(
                "task_progress",
                data=[json.dumps(p) for p in d["task_progress"]],
                dtype=h5py.string_dtype(encoding="utf-8"),
            )
            g.create_dataset("states", data=np.stack(d["states"]).astype(np.float64))
            actions = np.stack(d["actions"]).astype(np.float32)
            absolute = np.stack(d["actions_abs"]).astype(np.float32)
            g.create_dataset("actions", data=actions)
            g.create_dataset("actions_abs", data=absolute)
            ad = g.create_group("action_dict")
            ad.create_dataset("rel_pos", data=actions[:, :3])
            ad.create_dataset("rel_rot_axis_angle", data=actions[:, 3:6])
            ad.create_dataset("rel_rot_6d", data=_rot6d(actions[:, 3:6]))
            ad.create_dataset("abs_pos", data=absolute[:, :3])
            ad.create_dataset("abs_rot_axis_angle", data=absolute[:, 3:6])
            ad.create_dataset("abs_rot_6d", data=_rot6d(absolute[:, 3:6]))
            ad.create_dataset("gripper", data=actions[:, 6:7])
            g.create_dataset("rewards", data=d["rewards"])
            dones = np.zeros(n, dtype=np.int64)
            dones[-1] = 1
            g.create_dataset("dones", data=dones)
            obs, nxt = g.create_group("obs"), g.create_group("next_obs")
            for key in d["obs"][0]:
                obs.create_dataset(key, data=np.stack([o[key] for o in d["obs"]]))
                nxt.create_dataset(key, data=np.stack([o[key] for o in d["next_obs"]]))
        f.create_group("mask")  # filter keys, which the recorder leaves empty


def _as_old(new: Path, path: Path, *, version: int = 3, mask=True) -> None:
    """The same steps as the recorder wrote them up to ``version``: obs[t]
    after actions[t], states[t] after it (the last one, never in a format 4
    file, NaN: realigning drops it), next_obs[t] = obs[t + 1], the last
    repeated. With ``mask``, filter keys as a user may add them."""
    shutil.copy(new, path)
    with h5py.File(path, "r+") as f:
        env_args = json.loads(f["data"].attrs["env_args"])
        env_args["env_kwargs"]["format_version"] = version
        f["data"].attrs["env_args"] = json.dumps(env_args)
        for g in f["data"].values():
            after = {k: v[()] for k, v in g["next_obs"].items()}
            for k, v in after.items():
                g["obs"][k][...] = v
                g["next_obs"][k][...] = np.concatenate([v[1:], v[-1:]])
            st = g["states"][()]
            g["states"][...] = np.concatenate([st[1:], np.full_like(st[:1], np.nan)])
            for key in ("start_step", "reset_seed", "gaps"):  # format 4's
                if key in g.attrs:
                    del g.attrs[key]
        if mask:
            names = sorted(f["data"], key=lambda s: int(s.rpartition("_")[2]))
            f["mask"].create_dataset("train", data=np.array([n.encode() for n in names[:2]]))
            f["mask"].create_dataset("valid", data=np.array([n.encode() for n in names[1:]]))


def _relabel(path: Path, version: int) -> None:
    """Say ``version`` in env_kwargs, as an edit by hand would."""
    with h5py.File(path, "r+") as f:
        env_args = json.loads(f["data"].attrs["env_args"])
        env_args["env_kwargs"]["format_version"] = version
        env_args["env_kwargs"].pop("realigned_from_format_version", None)
        f["data"].attrs["env_args"] = json.dumps(env_args)


def _demo(f, name: str) -> dict:
    """A demo's attributes and every dataset by its path in the demo."""
    g = f["data"][name]
    out = {"attrs": dict(g.attrs)}
    g.visititems(lambda path, item: out.update({path: item[()]}) if _is_data(item) else None)
    return out


def _is_data(item) -> bool:
    return isinstance(item, h5py.Dataset)


def _action_side(demo: dict) -> list[str]:
    """The datasets of the action's row, which realigned_first/ keeps the
    first row of."""
    return [
        k
        for k in demo
        if k != "attrs" and k not in _BEFORE and not k.startswith(("obs/", "next_obs/"))
    ]


def _dtype(ds) -> tuple:
    """A dataset's type, a variable-length string's encoding included."""
    return ds.dtype.str, h5py.check_string_dtype(ds.dtype)


@needs_h5py
def test_an_old_recording_realigns_to_format_4_without_each_first_row(tmp_path):
    """Realigned, the old file is the format 4 file of the same steps,
    dataset by dataset, without each demo's first row, which is in the demo's
    realigned_first/ instead, and without the one-sample demo."""
    new, old, out = tmp_path / "new.hdf5", tmp_path / "old.hdf5", tmp_path / "out.hdf5"
    _format_4(new, _session())
    _as_old(new, old)
    with h5py.File(old, "r+") as f:
        # The attributes of a dataset go along with its rows.
        f["data/demo_0/obs/robot0_eef_pos"].attrs["units"] = "m"
        f["data/demo_0/actions"].attrs["units"] = "normalized"
        # Format 4's demo attributes, should an old file have them, would
        # count from the row that goes: they are not carried over.
        f["data/demo_2"].attrs["start_step"] = 7
        f["data/demo_2"].attrs["gaps"] = np.zeros(0, dtype=np.int64)
    report = realign_recording(old, out)
    assert report == {
        "format_version": 3,
        "demos": {"demo_0": 4, "demo_2": 2},
        "left_out": ["demo_1"],
    }
    with h5py.File(new, "r") as want, h5py.File(out, "r") as got:
        args_want = json.loads(want["data"].attrs["env_args"])
        args_got = json.loads(got["data"].attrs["env_args"])
        assert args_got["env_kwargs"].pop("realigned_from_format_version") == 3
        assert args_got == args_want and args_got["env_kwargs"]["format_version"] == 4
        # The one-sample demo has no row left: it is gone, from the masks too.
        assert list(got["data"]) == ["demo_0", "demo_2"] and got["data"].attrs["total"] == 6
        assert [x.decode() for x in got["mask/train"][()]] == ["demo_0"]
        assert [x.decode() for x in got["mask/valid"][()]] == ["demo_2"]
        assert got["mask/train"].dtype == np.dtype("S6")
        for name, paid in (("demo_0", 1.0), ("demo_2", 2.0)):
            a, b = _demo(want, name), _demo(got, name)
            first = {k: b.pop(k) for k in list(b) if k.startswith("realigned_first/")}
            # Format 4's datasets, each without its first row, and no others.
            assert sorted(a) == sorted(b), name
            for key, v in a.items():
                if key == "attrs":
                    continue
                np.testing.assert_array_equal(b[key], v[1:], err_msg=f"{name}/{key}")
                assert _dtype(got["data"][name][key]) == _dtype(want["data"][name][key]), key
            # The first action, which has no row now: its own rows, as recorded.
            assert sorted(first) == sorted(f"realigned_first/{k}" for k in _action_side(a))
            for key in _action_side(a):
                head = got["data"][name]["realigned_first"][key]
                np.testing.assert_array_equal(head[()], a[key][:1], err_msg=f"{name}/{key}")
                assert _dtype(head) == _dtype(want["data"][name][key]), key
            assert b["attrs"]["num_samples"] == a["attrs"]["num_samples"] - 1
            assert b["attrs"]["control_mode"] == a["attrs"]["control_mode"]
            # An old file does not say where a demo started or where it had gaps.
            assert not {"start_step", "reset_seed", "gaps"} & set(b["attrs"])
            assert set(a["attrs"]) - set(b["attrs"]) == {"start_step", "reset_seed", "gaps"}
            # paid_at_start takes in the points of the row that moved out.
            before = json.loads(a["attrs"]["episode_summary"])
            after = json.loads(b["attrs"]["episode_summary"])
            assert after["paid_at_start"] == paid
            assert {k: v for k, v in after.items() if k != "paid_at_start"} == {
                k: v for k, v in before.items() if k != "paid_at_start"
            }
            reward = after["task_progress"]["reward"]
            assert math.isclose(
                float(b["rewards"].sum()) * reward["max"],
                reward["earned"] - reward["penalty"] + reward["shaping"]["return"] - paid,
                abs_tol=1e-12,
            )
        demo_0 = got["data/demo_0"]
        assert demo_0["obs/robot0_eef_pos"].attrs["units"] == "m"
        assert demo_0["actions"].attrs["units"] == "normalized"
        assert demo_0["realigned_first/actions"].attrs["units"] == "normalized"
        # The final observation stays, as the last next_obs; the state after
        # the last action (NaN here) is not in format 4.
        assert not np.isnan(demo_0["states"][()]).any()
        np.testing.assert_array_equal(
            demo_0["next_obs/robot0_eef_pos"][-1], want["data/demo_0/next_obs/robot0_eef_pos"][-1]
        )


@needs_h5py
def test_a_realigned_demo_replays_from_the_reset_through_its_first_action(tmp_path):
    """A demo recorded from a reset (seed 0, as the editor's), realigned: it
    starts one action later, and realigned_first/actions[0] takes the world
    from that reset to its states[0]; its own actions take it on to each next
    row and to the final observation. Without realigned_first nothing in the
    file got there."""
    new, old, out = tmp_path / "new.hdf5", tmp_path / "old.hdf5", tmp_path / "out.hdf5"
    _format_4(new, {"demo_0": _steps(0, 6)})
    _as_old(new, old, mask=False)
    realign_recording(old, out)
    world = _World()
    with h5py.File(out, "r") as f:
        g = f["data/demo_0"]
        assert g.attrs["num_samples"] == 5
        assert not np.array_equal(world.reset(0), g["states"][0])
        np.testing.assert_array_equal(world.step(g["realigned_first/actions"][0]), g["states"][0])
        for key, value in _obs(world.state, True).items():
            np.testing.assert_array_equal(g["obs"][key][0], value, err_msg=key)
        for t, action in enumerate(g["actions"][()]):
            world.step(action)
            if t + 1 < len(g["states"]):
                np.testing.assert_array_equal(world.state, g["states"][t + 1], err_msg=f"{t}")
            for key, value in _obs(world.state, True).items():
                np.testing.assert_array_equal(g["next_obs"][key][t], value, err_msg=f"{t} {key}")


@needs_h5py
def test_format_1_task_progress_and_files_without_a_version_realign_too(tmp_path):
    """Format 1 kept task_progress as a JSON attribute and had no
    episode_summary, and files from before format 1 have no format_version
    (read as 0). A scene without movable objects records an empty object."""
    new, old, out = tmp_path / "new.hdf5", tmp_path / "old.hdf5", tmp_path / "out.hdf5"
    _format_4(new, {"demo_0": _steps(3, 4, objects=False)})
    _as_old(new, old, version=1, mask=False)
    with h5py.File(old, "r+") as f:
        env_args = json.loads(f["data"].attrs["env_args"])
        del env_args["env_kwargs"]["format_version"]
        f["data"].attrs["env_args"] = json.dumps(env_args)
        g = f["data/demo_0"]
        steps = [json.loads(p) for p in g["task_progress"].asstr()[()]]
        del g["task_progress"]
        del g.attrs["episode_summary"]
        g.attrs["task_progress"] = json.dumps(steps)
        assert g["obs/object"].shape == (4, 0)
    assert realign_recording(old, out) == {
        "format_version": 0,
        "demos": {"demo_0": 3},
        "left_out": [],
    }
    with h5py.File(out, "r") as f:
        kw = json.loads(f["data"].attrs["env_args"])["env_kwargs"]
        assert kw["format_version"] == 4 and kw["realigned_from_format_version"] == 0
        g = f["data/demo_0"]
        assert json.loads(g.attrs["task_progress"]) == steps[1:]
        assert json.loads(g["realigned_first"].attrs["task_progress"]) == steps[:1]
        assert "task_progress" not in g and "episode_summary" not in g.attrs
        assert len(g["actions"]) == 3 and g["obs/object"].shape == (3, 0)
        assert list(f["mask"]) == []


@needs_h5py
def test_the_helper_refuses_what_it_must_not_rewrite(tmp_path):
    """A file of format 4 or later, one whose rows are in format 4's order
    already though it says 3, and one the old recorder did not write: none is
    rewritten, and no file is written. Nor is the old file, or another that
    exists, unless overwrite=True."""
    new, old = tmp_path / "new.hdf5", tmp_path / "old.hdf5"
    _format_4(new, _session())
    _as_old(new, old)
    for version in (4, 5):  # in order already
        current = tmp_path / f"format_{version}.hdf5"
        shutil.copy(new, current)
        _relabel(current, version)
        with pytest.raises(ValueError, match=f"is format {version}: its obs and states {_DONE}"):
            realign_recording(current, tmp_path / "x.hdf5")
    # A format 4 file labelled 3, and an old file realigned already that
    # still says 3: shifting either again would put each row a step early.
    labelled, realigned, half = (tmp_path / f"{n}.hdf5" for n in ("labelled", "again", "half"))
    shutil.copy(new, labelled)
    realign_recording(old, realigned)
    shutil.copy(realigned, half)
    with h5py.File(half, "r+") as f:  # the slicing alone, num_samples as it was
        for g in f["data"].values():
            del g["realigned_first"]
            g.attrs["num_samples"] = int(g.attrs["num_samples"]) + 1
    for path, why in (
        (labelled, "demo_0/next_obs/object has a last row of its own: "),
        (realigned, "demo_0 has realigned_first/: "),
        (half, "demo_0 has 4 rows of actions and num_samples 5, which the recorder never wrote: "),
    ):
        _relabel(path, 3)
        with pytest.raises(ValueError) as raised:
            realign_recording(path, tmp_path / "y.hdf5")
        assert f"{why}the file says format 3, but {_AGAIN}" in str(raised.value), path.name
    # One whose next_obs is no shift of obs at all.
    scrambled = tmp_path / "scrambled.hdf5"
    shutil.copy(old, scrambled)
    with h5py.File(scrambled, "r+") as f:
        f["data/demo_0/next_obs/robot0_eef_pos"][0] += 1.0
    with pytest.raises(ValueError, match=f"demo_0/next_obs/robot0_eef_pos {_NOT_OLD}"):
        realign_recording(scrambled, tmp_path / "y.hdf5")
    assert not (tmp_path / "x.hdf5").exists() and not (tmp_path / "y.hdf5").exists()
    # Never over the old file, nor over another unless asked.
    with pytest.raises(ValueError, match="not over it"):
        realign_recording(old, old)
    with pytest.raises(ValueError, match="not over it"):  # the same file, spelled otherwise
        realign_recording(old, os.path.join(old.parent, ".", old.name))
    taken = tmp_path / "taken.hdf5"
    taken.write_bytes(b"keep")
    with pytest.raises(FileExistsError, match="overwrite=True"):
        realign_recording(old, taken)
    assert taken.read_bytes() == b"keep"
    assert realign_recording(old, taken, overwrite=True)["demos"] == {"demo_0": 4, "demo_2": 2}
    with h5py.File(taken, "r") as f:
        assert f["data/demo_0"].attrs["num_samples"] == 4


def _command(*args, code=None) -> subprocess.CompletedProcess:
    """``python -m world_engine_client.recording_realign`` (or ``code``) in
    a process of its own, with this package on its path."""
    src = Path(recording_realign.__file__).resolve().parents[1]
    run = ["-c", code] if code else ["-m", "world_engine_client.recording_realign"]
    return subprocess.run(
        [sys.executable, *run, *map(str, args)],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(src)},
        timeout=120,
    )


@needs_h5py
def test_the_command_line(tmp_path):
    """The command realigns the file, says that each demo starts one action
    later and which demos it left out (in the order they were recorded), and
    writes over nothing."""
    new, old, out = tmp_path / "new.hdf5", tmp_path / "old.hdf5", tmp_path / "out.hdf5"
    demos = {f"demo_{k}": _steps(k, rows) for k, rows in ((0, 4), (1, 3), (2, 1), (10, 1))}
    _format_4(new, demos)
    _as_old(new, old)
    done = _command(old, out)
    assert done.returncode == 0, done.stderr
    assert done.stdout.startswith(f"{out}: 2 demo(s) of format 3 in format 4's layout, 5 samples.")
    assert "Each demo now starts one action later" in done.stdout
    assert "step realigned_first/actions[0], and the world is at states[0]" in done.stdout
    assert done.stdout.endswith("Left out, one sample and none left: demo_2, demo_10\n")
    with h5py.File(out, "r") as f:
        assert {k: int(g.attrs["num_samples"]) for k, g in f["data"].items()} == {
            "demo_0": 3,
            "demo_1": 2,
        }
        assert [x.decode() for x in f["mask/valid"][()]] == ["demo_1"]
    written = out.read_bytes()
    # The destination exists: the command line has no overwrite=True to offer.
    taken = _command(old, out)
    assert taken.returncode == 1 and "overwrite" not in taken.stderr
    assert taken.stderr == f"not realigned: {out} exists; remove it or name another file\n"
    assert out.read_bytes() == written
    again = _command(out, tmp_path / "z.hdf5")
    assert again.returncode == 1 and again.stderr.startswith("not realigned: ")
    assert _DONE in again.stderr and not (tmp_path / "z.hdf5").exists()
    text = tmp_path / "notes.txt"
    text.write_text("not HDF5")
    for bad in (text, tmp_path / "missing.hdf5"):
        failed = _command(bad, tmp_path / "z.hdf5")
        assert failed.returncode == 1 and failed.stderr.startswith("not realigned: ")
    usage = _command(old)
    assert usage.returncode == 2
    assert usage.stderr == (
        "usage: python -m world_engine_client.recording_realign OLD.hdf5 NEW.hdf5\n"
    )


def test_without_h5py_the_helper_says_to_install_it(tmp_path, monkeypatch, capsys):
    """h5py is not a dependency of the package: without it, the function and
    the command say how to install it, before they look at any file. Runs
    with h5py installed too, which it hides."""
    old, new = tmp_path / "old.hdf5", tmp_path / "new.hdf5"
    message = (
        "realigning a recording needs h5py, which world-engine-client does not install: "
        "pip install h5py"
    )
    monkeypatch.setitem(sys.modules, "h5py", None)
    with pytest.raises(ImportError, match=message) as raised:
        realign_recording(old, new)
    assert raised.value.name == "h5py"
    assert recording_realign.main([str(old), str(new)]) == 1
    assert capsys.readouterr().err == f"not realigned: {message}\n"
    code = (
        "import runpy, sys\n"
        "sys.modules['h5py'] = None\n"
        "runpy.run_module('world_engine_client.recording_realign', run_name='__main__',"
        " alter_sys=True)\n"
    )
    done = _command(old, new, code=code)
    assert done.returncode == 1 and done.stderr == f"not realigned: {message}\n"
    assert not new.exists()


def test_an_h5py_that_does_not_import_raises_its_own_error(monkeypatch):
    """h5py installed but broken (built for another numpy, say) is not a
    missing h5py: its own error comes through, not the advice to install it."""

    class Broken:
        def find_spec(self, name, path=None, target=None):
            if name == "h5py":
                raise ImportError("h5py is built for another numpy")
            return None

    monkeypatch.delitem(sys.modules, "h5py", raising=False)
    monkeypatch.setattr(sys, "meta_path", [Broken(), *sys.meta_path])
    with pytest.raises(ImportError, match="built for another numpy") as raised:
        realign_recording("old.hdf5", "new.hdf5")
    assert "pip install h5py" not in str(raised.value)
