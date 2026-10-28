"""
Simulated Mitsubishi RV-6S, drop-in replacement of ``ctu_mitsubishi.Rv6s``
and ``ctu_mitsubishi.Rv6sGripper``.

The interface and the kinematics (DH parameters, joint limits, ``fk``,
``ik``) are taken from ``ctu_mitsubishi`` 0.1.5 (BSD 3-Clause, Copyright (c)
2023, CTURobotics, see LICENSE-ctu_mitsubishi), so the simulator returns
exactly the same poses and IK solutions as the real robot. Methods the real
class does not have (``in_motion``, ``set_speed``, ``release``, ...) are not
here either, code that runs in the simulator runs on the robot.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike

from .sim_robot import SimRobot

URDF_PATH_RV6S = Path(__file__).parent / "urdf" / "RV6S.urdf"

# Rated maximum joint speeds of the RV-6S [deg/s] at the 25 % override
# ctu_mitsubishi sets in initialize() (estimate, not measured on the robot)
_MAX_SPEED_DEG = np.array([401, 321, 401, 352, 450, 660]) * 0.25


def circle_circle_intersection(
    c0: ArrayLike, r0: float, c1: ArrayLike, r1: float
) -> list[np.ndarray]:
    """Computes intersection of the circles defined by center c_i and radius r_i.
    Returns empty array if there is no solution, two solutions otherwise. If there
    are infinite number of solutions, select two (almost) randomly.
    """
    c0 = np.asarray(c0)
    c1 = np.asarray(c1)
    d = np.linalg.norm(c1 - c0)
    if r0 + r1 < d < np.abs(r0 - r1) or d > r0 + r1:
        return []

    if np.isclose(d, 0) and np.isclose(r0, r1):
        return [
            c0 + [r0 * np.cos(a), r0 * np.sin(a)]
            for a in np.random.uniform(-np.pi, np.pi, 2)
        ]
    a = (r0**2 - r1**2 + d**2) / (2 * d)
    assert r0**2 - a**2 >= 0, "Invalid circle_circle_intersection"

    h = np.sqrt(r0**2 - a**2)
    x0, y0 = c0
    x1, y1 = c1
    x2 = x0 + a * (x1 - x0) / d
    y2 = y0 + a * (y1 - y0) / d
    x3 = x2 + h * (y1 - y0) / d
    y3 = y2 - h * (x1 - x0) / d
    x4 = x2 - h * (y1 - y0) / d
    y4 = y2 + h * (x1 - x0) / d

    return [np.array([x3, y3]), np.array([x4, y4])]


class Rv6s(SimRobot):
    """
    Simulated RV-6S. Accepts the same arguments as ``ctu_mitsubishi.Rv6s``
    plus the keyword-only simulator options of
    :class:`crs_sim.sim_robot.SimRobot` (``visualize``, ``speedup``,
    ``camera``, ``on_contact``, ...). The port is not opened.
    """
    def __init__(self, port="/dev/ttyUSB_Robot", baudrate=38400, debug=False,
                 **sim_kwargs) -> None:
        self._debug = debug
        self._initialized = False
        self.q_home = np.deg2rad([0, 0, 90, 0, 90, 0])
        self.q_min = np.deg2rad([-170, -92, -107, -160, -120, -360])
        self.q_max = np.deg2rad([170, 135, 166, 160, 120, 360])

        self.dh_theta_off = np.deg2rad([0, -90, -90, 0, 0, 180])
        self.dh_a = np.array([85, 280, 100, 0, 0, 0]) / 1000.0
        self.dh_d = np.array([350, 0, 0, 315, 0, 85]) / 1000.0
        self.dh_alpha = np.deg2rad([-90, 0, -90, 90, -90, 0])

        self._last_q = None
        self._max_velocity = np.deg2rad(_MAX_SPEED_DEG)
        sim_kwargs.setdefault("urdf_path", URDF_PATH_RV6S)
        super().__init__(**sim_kwargs)

    # ------------------------------------------------------------------
    # Interface of ctu_mitsubishi.Rv6s
    # ------------------------------------------------------------------
    def close_connection(self):
        """Close the connection to the robot's control unit."""
        if self._initialized:
            self.stop_robot()

    def stop_robot(self):
        """Stop the robot and perform a safe shutdown."""
        if self._initialized:
            self._start_motion(self._current_q())
            self._initialized = False

    def initialize(self):
        """Performs robot initialization (the robot does not move)."""
        self._initialized = True

    def get_q(self) -> np.ndarray:
        """Get current joint configuration [rad]."""
        assert self._initialized, "You need to initialize the robot before moving it."
        return self._current_q()

    def move_to_q(self, q: ArrayLike):
        """Move robot to the given joint configuration [rad] using coordinated movement.
        Initialization has be called before to set up the robot."""
        assert self._initialized, "You need to initialize the robot before moving it."
        assert self.in_limits(q), "Joint limits violated."
        self._last_q = np.asarray(q).copy()
        self._raise_contacts()
        self._start_motion(q)

    def soft_home(self):
        """Move robot to the home position using move to q function."""
        self.move_to_q(self.q_home)

    def in_limits(self, q: ArrayLike) -> bool:
        """Return whether the given joint configuration is in joint limits."""
        return np.all(q >= self.q_min) and np.all(q <= self.q_max)

    @staticmethod
    def _rx(angle: float) -> np.ndarray:
        """Return SE3 transformation that rotates around x-axis."""
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]])

    @staticmethod
    def _ry(angle: float) -> np.ndarray:
        """Return SE3 transformation that rotates around x-axis."""
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]])

    @staticmethod
    def _rz(angle: float) -> np.ndarray:
        """Return SE3 transformation that rotates around z-axis."""
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    @staticmethod
    def _t(tx: float = 0.0, ty: float = 0.0, tz: float = 0.0) -> np.ndarray:
        """Return SE3 transformation that translates along x-axis."""
        return np.array([[1, 0, 0, tx], [0, 1, 0, ty], [0, 0, 1, tz], [0, 0, 0, 1]])

    @staticmethod
    def dh_to_se3(d: float, theta: float, a: float, alpha: float) -> np.ndarray:
        """Compute SE3 matrix from DH parameters."""
        return (
            Rv6s._t(tz=d)
            @ Rv6s._rz(angle=theta)
            @ Rv6s._t(tx=a)
            @ Rv6s._rx(angle=alpha)
        )

    def fk(self, q: ArrayLike) -> np.ndarray:
        """Compute forward kinematics for the given joint configuration [rad].
        Return pose of the end-effector in the base frame. Homogeneous transformation
        matrix (4x4) is returned. The pose represents the flange pose w.r.t. base
        of the robot."""
        pose = np.eye(4)
        for d, a, alpha, theta, qi in zip(
            self.dh_d, self.dh_a, self.dh_alpha, self.dh_theta_off, q
        ):
            pose = pose @ self.dh_to_se3(d, qi + theta, a, alpha)
        return pose

    def _ik_5th_joint_pos(self, position: ArrayLike) -> list[np.ndarray]:
        """Compute inverse kinematics for the given position of the 5th joint
        (flange w.r.t. base) in the base frame.
        Return list of joint configurations [rad] for the first three joints.
        """
        position = np.asarray(position)
        assert position.shape == (3,), "Position must be 3D vector."
        x, y, z = position

        res = []
        j1s = [np.atan2(y, x), np.atan2(-y, -x)]
        for j1 in j1s:
            pose_base_j2 = self.dh_to_se3(
                self.dh_d[0], j1, self.dh_a[0], self.dh_alpha[0]
            ) @ self._rz(-np.pi / 2)
            xy_j2 = (np.linalg.inv(pose_base_j2) @ np.array([x, y, z, 1]))[:2]
            r1 = self.dh_a[1]
            r2 = np.linalg.norm([self.dh_d[3], self.dh_a[2]])
            sols = circle_circle_intersection([0, 0], r1, xy_j2, r2)
            j2s = [np.arctan2(s[1], s[0]) for s in sols]

            for j2 in j2s:
                ang = np.atan2(self.dh_a[2], self.dh_d[3])
                pose_j2_j3 = self._rz(j2) @ self._t(tx=r1) @ self._rz(-ang)
                xy_j3 = (
                    np.linalg.inv(pose_j2_j3) @ np.array([xy_j2[0], xy_j2[1], 0, 1])
                )[:2]
                j3 = np.atan2(xy_j3[1], xy_j3[0])
                res.append(np.array([j1, j2, j3]))
        return res

    def ik(self, pose: ArrayLike) -> list[np.ndarray]:
        """Compute inverse kinematics for the given pose of the end-effector (flange
        w.r.t. base) in the base frame. Homogeneous transformation matrix (4x4) is
        expected. Only in limits solutions are returned.
        Return list of joint configurations [rad].
        """
        pose = np.asarray(pose)
        assert pose.shape == (4, 4), "Pose must be 4x4 matrix."

        t_5th_joint = (pose @ self._t(tz=-self.dh_d[-1]))[:3, 3]
        q3s = self._ik_5th_joint_pos(t_5th_joint)

        res = []
        for q3 in q3s:
            q = np.zeros(6)
            q[:3] = q3
            pose0 = self.fk(q)  # pose with last 3 joints set to zero
            delta = pose0[:3, :3].T @ pose[:3, :3]
            for out in self.rotation_matrix_to_euler_zyz(delta):
                q[3:] = out
                res.append(q.copy())

                # add periodic solutions for the last joint
                res.append(q.copy())
                res[-1][5] += 2 * np.pi
                res.append(q.copy())
                res[-1][5] -= 2 * np.pi

        return [s for s in res if self.in_limits(s)]

    @staticmethod
    def rotation_matrix_to_euler_zyz(R):
        """
        Extract Euler angles (ZYZ convention) from a rotation matrix.

        Parameters:
            R (ndarray): 3x3 rotation matrix.

        Returns:
            two solutions, 2 tuples: Euler angles (alpha, beta, gamma) in radians.
        """
        beta = np.arccos(R[2, 2])  # Rotation around the new Y-axis

        if np.isclose(beta, 0):  # Handle gimbal lock (beta ≈ 0)
            alpha = np.arctan2(R[1, 0], R[0, 0])
            gamma = 0
            return [alpha, beta, gamma]
        elif np.isclose(beta, np.pi):  # Handle gimbal lock (beta ≈ π)
            alpha = np.arctan2(-R[1, 0], -R[0, 0])
            gamma = 0
            return [alpha, beta, gamma]

        alpha = np.arctan2(R[1, 2], R[0, 2])  # Rotation around Z-axis
        gamma = np.arctan2(R[2, 1], -R[2, 0])  # Rotation around the new Z-axis

        beta2 = 2 * np.pi - beta
        beta2 = np.atan2(np.sin(beta2), np.cos(beta2))
        alpha2 = np.arctan2(-R[1, 2], -R[0, 2])
        gamma2 = np.arctan2(-R[2, 1], R[2, 0])

        return [alpha, beta, gamma], [alpha2, beta2, gamma2]

    def wait_for_motion_stop(
        self, timeout: float = 10, poll_interval: float = 0.1, tol: float = 0.0087
    ) -> bool:
        """Wait for the robot to stop moving.
        Args:
            timeout: Maximum time to wait for the robot to stop moving.
            poll_interval: Time between checking the robot's state.
            tol: Tolerance for the robot's joint positions to be considered stopped
             [rad].
        Returns:
            True if the robot has stopped moving in a given time, False otherwise.
        """
        assert self._initialized, "You need to initialize the robot before moving it."
        return self._wait_for_motion(timeout)


class Rv6sGripper:
    """Simulated ``ctu_mitsubishi.Rv6sGripper``: remembers the last command,
    it is not rendered."""
    def __init__(self, dev="/dev/ttyUSB_Quido"):
        #: "open", "closed" or None before the first command
        self.state = None

    def open(self, duration: float = 1.0):
        """Start opening the gripper and stop after duration in seconds."""
        self.state = "open"

    def close(self, duration: float = 1.0):
        """Start closing the gripper and stop after duration in seconds."""
        self.state = "closed"

    def disconnect(self):
        """Stop controlling the gripper."""
