"""Realign a demonstration HDF5 of format 3 or older to format 4's layout.

The editor's Download Demonstrations writes robomimic's layout. In every file
whose format_version is 3 or lower, or that has none, the recorder stored, in
row t, the state and the observation that ``actions[t]`` left: ``obs[t]`` and
``states[t]`` came one step after the action, the observation the first
action was chosen from was never written, and ``next_obs[t]`` was
``obs[t + 1]``, the last row repeated. Format 4 stores what robomimic and
robosuite do: ``states[t]`` and ``obs[t]`` before ``actions[t]``,
``next_obs[t]`` after it. A file says which one it is in
``data.attrs["env_args"]["env_kwargs"]["format_version"]``.

An old demo of N rows has the state and observation after each of its N
actions. Shifted by one they make N - 1 rows of format 4: row j holds the old
``obs[j]`` and ``states[j]`` (what action j + 1 was chosen from),
``next_obs`` = the old ``obs[j + 1]``, and action j + 1's ``actions``,
``actions_abs``, ``action_dict``, ``rewards``, ``dones`` and
``task_progress``. The first action gets no row: the state and observation it
was chosen from were never recorded. So a realigned demo starts one action
after where it was recorded, and keeps that action in ``realigned_first/``:
the first row of each of those datasets, under the same names. For a demo
recorded from a reset, reset as then (seed 0 in the editor), step
``realigned_first/actions[0]``, and the world is at ``states[0]``. The old
last ``states`` row, the state after the last action, has no place in format
4, which keeps no state after the last action (robomimic does not either);
the final observation stays, as the last ``next_obs``.

A demo of one sample has no row left and is left out; the ``mask`` lists lose
its name. For a staged reward, ``episode_summary.paid_at_start`` grows by the
moved reward times the reward's ``max``, so ``sum(rewards) * max`` still
equals ``earned - penalty + shaping.return - paid_at_start``. The rest is
copied as it was. ``env_kwargs`` gets ``format_version`` 4 and
``realigned_from_format_version``, the version the file was recorded in (0
when it had none), which says what its other fields hold. Format 4's
``start_step``, ``reset_seed`` and ``gaps`` are not written: an old file does
not say where a demo started or whether the world moved between two rows.
The README's "Demonstration recordings" says how to tell an old file and
what to run.

Needs numpy and h5py. This package does not install h5py
(``pip install h5py``); without it, :func:`realign_recording` raises
ImportError and the command says so. From a shell:

    python -m world_engine_client.recording_realign old.hdf5 new.hdf5
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

# The first version whose obs and states come before the action.
ALIGNED_FORMAT_VERSION = 4
# A demo's datasets read before the action; every other per-step dataset
# belongs to the action of its row.
_BEFORE = ("states",)
# Each realigned demo's group for its first action's rows (module docstring).
FIRST = "realigned_first"
# Format 4's demo attributes that an old file cannot fill in.
_NOT_KNOWN = ("start_step", "reset_seed", "gaps")


def realign_recording(src, dst, *, overwrite: bool = False) -> dict:
    """Write ``src`` (format 3 or older) to ``dst`` in format 4's layout.

    Returns ``{"format_version", "demos": {name: samples}, "left_out":
    [names]}``: the version ``src`` was recorded in, the samples each demo
    kept (one fewer than it had) and the demos with nothing left. Raises
    ValueError for a file already in format 4 or later, or not laid out as
    the old recorder wrote it, and FileExistsError when ``dst`` exists,
    unless ``overwrite``; never writes over ``src``. Raises ImportError,
    before it reads anything, when h5py is not installed."""
    h5py = _h5py()

    if os.path.abspath(src) == os.path.abspath(dst):
        raise ValueError("write the realigned file next to the old one, not over it")
    if os.path.exists(dst) and not overwrite:
        raise FileExistsError(f"{dst} exists; pass overwrite=True to replace it")
    with h5py.File(src, "r") as fin:
        env_args = json.loads(_text(fin["data"].attrs["env_args"]))
        kwargs = env_args.setdefault("env_kwargs", {})
        version = int(kwargs.get("format_version", 0))
        if version >= ALIGNED_FORMAT_VERSION:
            raise ValueError(
                f"{src} is format {version}: its obs and states already come before each action"
            )
        for name in fin["data"]:
            _check_old_layout(fin["data"][name], name, version)
        report: dict = {"format_version": version, "demos": {}, "left_out": []}
        with h5py.File(dst, "w") as fout:
            _copy_attrs(fin, fout)
            data_in, data_out = fin["data"], fout.create_group("data")
            _copy_attrs(data_in, data_out)
            for name in sorted(data_in, key=_demo_order):
                demo = data_in[name]
                n = int(demo.attrs["num_samples"])
                if n < 2:
                    report["left_out"].append(name)
                    continue
                _realign_demo(demo, data_out.create_group(name), n)
                report["demos"][name] = n - 1
            data_out.attrs["total"] = int(sum(report["demos"].values()))
            kwargs["format_version"] = ALIGNED_FORMAT_VERSION
            kwargs["realigned_from_format_version"] = version
            data_out.attrs["env_args"] = json.dumps(env_args)
            for key in fin:
                if key == "data":
                    continue
                if key == "mask":
                    _copy_mask(fin[key], fout.create_group(key), set(report["left_out"]))
                else:
                    fin.copy(fin[key], fout, name=key)
    return report


def _check_old_layout(demo, name: str, version: int) -> None:
    """The old recorder wrote next_obs as obs shifted by one, the last row
    repeated, and as many rows as num_samples: a file that does not is not
    one of its recordings. One with realigned_first/, or whose next_obs is
    obs shifted by one with a new last row, is in format 4's layout already
    (realigned by hand with its format_version left as it was, say), and
    shifting it again would put every row one step early."""
    again = (
        f"the file says format {version}, but it was realigned already (by hand, with "
        "format_version left as it was?). Realign the file as it was downloaded"
    )
    if FIRST in demo:
        raise ValueError(f"{name} has {FIRST}/: {again}")
    rows = len(demo["actions"]) if "actions" in demo else None
    if rows is not None and rows != int(demo.attrs.get("num_samples", rows)):
        raise ValueError(
            f"{name} has {rows} rows of actions and num_samples {demo.attrs['num_samples']}, "
            f"which the recorder never wrote: {again}"
        )
    if "obs" not in demo or "next_obs" not in demo:
        return
    for key, ds in demo["obs"].items():
        if key not in demo["next_obs"]:
            continue
        obs, nxt = ds[()], demo["next_obs"][key][()]
        if not len(obs) or np.array_equal(nxt, np.concatenate([obs[1:], obs[-1:]])):
            continue
        if len(nxt) == len(obs) and np.array_equal(nxt[:-1], obs[1:]):
            raise ValueError(f"{name}/next_obs/{key} has a last row of its own: {again}")
        raise ValueError(
            f"{name}/next_obs/{key} is not obs shifted by one step, as the recorder "
            "wrote it up to format 3; this file is not one to realign"
        )


def _realign_demo(demo, out, n: int) -> None:
    first = out.create_group(FIRST)  # the first action's rows, which get no row
    for key, item in demo.items():
        if key == "next_obs":
            continue  # rebuilt from obs
        if key == "obs":
            obs, nxt = out.create_group("obs"), out.create_group("next_obs")
            for k, ds in item.items():
                _put(obs, k, ds, ds[: n - 1])
                _put(nxt, k, ds, ds[1:n])
        elif key in _BEFORE:
            _put(out, key, item, item[: n - 1])
        else:
            _copy_after(item, out, first, key, n)
    _copy_attrs(demo, out)
    for key in _NOT_KNOWN:
        if key in out.attrs:
            del out.attrs[key]
    out.attrs["num_samples"] = n - 1
    if "task_progress" in demo.attrs:  # format 1 and older: a JSON list attribute
        steps = json.loads(_text(demo.attrs["task_progress"]))
        if isinstance(steps, list) and len(steps) == n:
            out.attrs["task_progress"] = json.dumps(steps[1:])
            first.attrs["task_progress"] = json.dumps(steps[:1])
    if "episode_summary" in demo.attrs and "rewards" in demo:
        summary = json.loads(_text(demo.attrs["episode_summary"]))
        paid = summary.get("paid_at_start") if isinstance(summary, dict) else None
        reward = ((summary or {}).get("task_progress") or {}).get("reward") or {}
        if paid is not None and reward.get("max"):
            summary["paid_at_start"] = float(paid) + float(demo["rewards"][0]) * float(
                reward["max"]
            )
            out.attrs["episode_summary"] = json.dumps(summary)


def _copy_after(item, parent, first, key: str, n: int) -> None:
    """A dataset of the action's row: its rows after the first under
    ``parent``, the first under ``first`` (realigned_first); a group
    (action_dict) the same for each of its datasets. Anything not N long is
    copied as it was."""
    import h5py

    if isinstance(item, h5py.Group):
        group, head = parent.create_group(key), first.create_group(key)
        _copy_attrs(item, group)
        _copy_attrs(item, head)
        for k, child in item.items():
            _copy_after(child, group, head, k, n)
    elif item.shape and item.shape[0] == n:
        _put(parent, key, item, item[1:n])
        _put(first, key, item, item[:1])
    else:
        parent.copy(item, parent, name=key)


def _put(parent, key: str, like, data) -> None:
    ds = parent.create_dataset(key, data=data, dtype=like.dtype)
    _copy_attrs(like, ds)


def _copy_attrs(src, dst) -> None:
    for k, v in src.attrs.items():
        dst.attrs[k] = v


def _copy_mask(src, dst, left_out: set[str]) -> None:
    """robomimic filter keys without the demos left out."""
    _copy_attrs(src, dst)
    for k, ds in src.items():
        names = [x for x in ds[()] if _text(x) not in left_out]
        dst.create_dataset(k, data=np.array(names, dtype=ds.dtype))


def _demo_order(name: str):
    head, _, tail = name.rpartition("_")
    return (0, int(tail), "") if head == "demo" and tail.isdigit() else (1, 0, name)


def _text(value) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _h5py():
    """h5py, which world-engine-client does not depend on. Without it, an
    ImportError that says how to install it; an h5py that is installed but
    does not import raises its own error."""
    try:
        import h5py
    except ImportError as exc:
        if exc.name != "h5py":
            raise
        raise ImportError(
            "realigning a recording needs h5py, which world-engine-client does not "
            "install: pip install h5py",
            name="h5py",
        ) from exc
    return h5py


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 2:
        print(
            "usage: python -m world_engine_client.recording_realign OLD.hdf5 NEW.hdf5",
            file=sys.stderr,
        )
        return 2
    try:
        report = realign_recording(args[0], args[1])
    except FileExistsError:
        print(f"not realigned: {args[1]} exists; remove it or name another file", file=sys.stderr)
        return 1
    except (ImportError, ValueError, OSError, KeyError) as exc:
        print(f"not realigned: {exc}", file=sys.stderr)
        return 1
    kept = report["demos"]
    print(
        f"{args[1]}: {len(kept)} demo(s) of format {report['format_version']} in format "
        f"{ALIGNED_FORMAT_VERSION}'s layout, {sum(kept.values())} samples. Each demo now "
        "starts one action later: the observation its first action was chosen from was "
        f"never recorded. That action is in {FIRST}/: for a demo recorded from a reset "
        f"(seed 0 in the editor), reset, step {FIRST}/actions[0], and the world is at "
        "states[0]."
    )
    if report["left_out"]:
        print(f"Left out, one sample and none left: {', '.join(report['left_out'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
