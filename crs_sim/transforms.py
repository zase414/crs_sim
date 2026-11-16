"""
Small numpy helpers for 4x4 homogeneous transforms.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def rx(angle: float) -> np.ndarray:
    """4x4 rotation about x [rad]."""
    c, s = np.cos(angle), np.sin(angle)
    t = np.eye(4)
    t[1:3, 1:3] = [[c, -s], [s, c]]
    return t


def ry(angle: float) -> np.ndarray:
    """4x4 rotation about y [rad]."""
    c, s = np.cos(angle), np.sin(angle)
    t = np.eye(4)
    t[[0, 0, 2, 2], [0, 2, 0, 2]] = [c, s, -s, c]
    return t


def rz(angle: float) -> np.ndarray:
    """4x4 rotation about z [rad]."""
    c, s = np.cos(angle), np.sin(angle)
    t = np.eye(4)
    t[:2, :2] = [[c, -s], [s, c]]
    return t


def translation(xyz: ArrayLike) -> np.ndarray:
    """4x4 pure translation [m]."""
    t = np.eye(4)
    t[:3, 3] = xyz
    return t


def pose(xyz: ArrayLike = (0.0, 0.0, 0.0), yaw: float = 0.0) -> np.ndarray:
    """4x4 pose from a position [m] and a rotation about z [rad]."""
    return translation(xyz) @ rz(yaw)


def so3_log(rot: np.ndarray) -> np.ndarray:
    """Rotation matrix -> rotation vector (axis * angle)."""
    cos = np.clip((np.trace(rot) - 1) / 2, -1.0, 1.0)
    angle = np.arccos(cos)
    if angle < 1e-9:
        return np.zeros(3)
    if np.pi - angle < 1e-6:
        # axis from the diagonal, sign fixed by the largest component
        axis = np.sqrt(np.maximum((np.diag(rot) + 1) / 2, 0))
        i = np.argmax(axis)
        axis = rot[:, i] + np.eye(3)[i]
        return angle * axis / np.linalg.norm(axis)
    w = np.array([rot[2, 1] - rot[1, 2], rot[0, 2] - rot[2, 0], rot[1, 0] - rot[0, 1]])
    return angle * w / (2 * np.sin(angle))


def so3_exp(vec: ArrayLike) -> np.ndarray:
    """Rotation vector (axis * angle) -> rotation matrix."""
    vec = np.asarray(vec, dtype=float)
    angle = np.linalg.norm(vec)
    if angle < 1e-12:
        return np.eye(3)
    k = vec / angle
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(angle) * kx + (1 - np.cos(angle)) * kx @ kx


def interpolate(a: np.ndarray, b: np.ndarray, s: float) -> np.ndarray:
    """Pose between a (s=0) and b (s=1): linear position, geodesic rotation."""
    t = np.eye(4)
    t[:3, 3] = (1 - s) * a[:3, 3] + s * b[:3, 3]
    t[:3, :3] = a[:3, :3] @ so3_exp(s * so3_log(a[:3, :3].T @ b[:3, :3]))
    return t


def densify(poses: ArrayLike, max_step: float = 0.005,
            max_angle: float = np.deg2rad(5)) -> np.ndarray:
    """
    Insert poses so that consecutive poses are at most max_step [m] and
    max_angle [rad] apart.
    :param poses: Nx4x4 poses
    :return: Mx4x4 poses, the original ones included
    """
    poses = np.asarray(poses, dtype=float)
    result = [poses[0]]
    for a, b in zip(poses[:-1], poses[1:]):
        dist = np.linalg.norm(b[:3, 3] - a[:3, 3])
        angle = np.linalg.norm(so3_log(a[:3, :3].T @ b[:3, :3]))
        n = max(1, int(np.ceil(max(dist / max_step, angle / max_angle))))
        result += [interpolate(a, b, s) for s in np.arange(1, n + 1) / n]
    return np.array(result)
