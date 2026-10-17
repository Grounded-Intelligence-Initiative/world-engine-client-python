"""Pure-numpy rotation helpers for dataset consumers.

World Engine demonstration HDF5s carry 6D rotations in ``action_dict/*_rot_6d``
using the robomimic convention: the FIRST TWO ROWS of the rotation matrix,
flattened; decoding is Gram-Schmidt (Zhou et al., CVPR 2019). These helpers
round-trip that encoding without a torch/robomimic dependency.
"""

from __future__ import annotations

import numpy as np


def matrix_to_rot6d(m: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix -> 6D (first two ROWS, flattened; robomimic)."""
    m = np.asarray(m, dtype=np.float64).reshape(3, 3)
    return m[:2, :].reshape(6)


def rot6d_to_matrix(r6: np.ndarray) -> np.ndarray:
    """6D -> 3x3 rotation matrix via Gram-Schmidt (robomimic convention)."""
    r6 = np.asarray(r6, dtype=np.float64).reshape(6)
    a1, a2 = r6[:3], r6[3:]
    b1 = a1 / max(1e-12, float(np.linalg.norm(a1)))
    a2p = a2 - np.dot(b1, a2) * b1
    b2 = a2p / max(1e-12, float(np.linalg.norm(a2p)))
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=0)


def axisangle_to_matrix(rv: np.ndarray) -> np.ndarray:
    """Axis-angle rotation vector (axis*angle, rad) -> 3x3 matrix (Rodrigues)."""
    rv = np.asarray(rv, dtype=np.float64).reshape(3)
    angle = float(np.linalg.norm(rv))
    if angle < 1e-12:
        return np.eye(3)
    k = rv / angle
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def matrix_to_axisangle(m: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix -> axis-angle rotation vector (|rv| <= pi)."""
    m = np.asarray(m, dtype=np.float64).reshape(3, 3)
    cos = np.clip((np.trace(m) - 1.0) / 2.0, -1.0, 1.0)
    angle = float(np.arccos(cos))
    if angle < 1e-12:
        return np.zeros(3)
    if abs(angle - np.pi) < 1e-6:
        # Near pi: extract the axis from the symmetric part.
        w = np.sqrt(np.maximum(0.0, (np.diag(m) + 1.0) / 2.0))
        axis = w / max(1e-12, float(np.linalg.norm(w)))
        # Fix signs from the off-diagonals.
        if m[0, 1] + m[1, 0] < 0:
            axis[1] = -axis[1]
        if m[0, 2] + m[2, 0] < 0:
            axis[2] = -axis[2]
        return axis * angle
    axis = np.array([m[2, 1] - m[1, 2], m[0, 2] - m[2, 0], m[1, 0] - m[0, 1]]) / (
        2.0 * np.sin(angle)
    )
    return axis * angle
