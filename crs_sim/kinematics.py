"""
DH forward / analytical inverse kinematics of the CRS robots.

The math is taken verbatim from ``ctu_crs.crs_robot`` (BSD 3-Clause,
Copyright (c) 2023, CTURobotics, see LICENSE-ctu_crs) so that the simulator
returns exactly the same poses and IK solutions as the real robot.
"""
from __future__ import annotations

import numpy as np
from numpy import allclose
from numpy.typing import ArrayLike


class DHKinematics:
    """
    Kinematic model of a 6 DoF CRS arm described by DH parameters.
    """
    def __init__(self, link_lengths: ArrayLike) -> None:
        self.link_lengths = np.asarray(link_lengths, dtype=float)
        self.dh_offset = np.deg2rad(np.array([0.0, -270.0, -90.0, 0.0, 0.0, 0.0]))
        self.dh_d = [
            self.link_lengths[0],
            0,
            0,
            self.link_lengths[2],
            0,
            self.link_lengths[3],
        ]
        self.dh_a = [0, self.link_lengths[1], 0, 0, 0, 0]
        self.dh_alpha = np.deg2rad(np.array([90.0, 0.0, 270.0, 90.0, 270.0, 0]))

    @staticmethod
    def dh_to_se3(d: float, theta: float, a: float, alpha: float) -> np.ndarray:
        """Compute SE3 matrix from DH parameters."""
        tz = np.eye(4)
        tz[2, 3] = d
        rz = np.eye(4)
        rz[:2, :2] = np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
        )
        tx = np.eye(4)
        tx[0, 3] = a
        rx = np.eye(4)
        rx[1:3, 1:3] = np.array(
            [[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]]
        )
        return tz @ rz @ tx @ rx

    def fk(self, q: ArrayLike) -> np.ndarray:
        """
        Forward kinematics.
        :param q: joint configuration [rad]
        :return: 4x4 homogeneous matrix of the flange w.r.t. the robot base
        """
        pose = np.eye(4)
        for d, a, alpha, theta, qi in zip(
            self.dh_d, self.dh_a, self.dh_alpha, self.dh_offset, q
        ):
            pose = pose @ self.dh_to_se3(d, qi + theta, a, alpha)
        return pose

    def fk_flange_pos(self, q: ArrayLike) -> np.ndarray:
        """
        3D position of the flange (wrist center) w.r.t. the robot base.
        :param q: joint configuration [rad]
        """
        return (self.fk(q) @ np.array([0, 0, -self.dh_d[-1], 1]))[:3]

    def _ik_flange_pos(
        self, flange_pos: np.ndarray, singularity_theta1: float = 0
    ) -> list[np.ndarray]:
        """Solve IK for the position of the flange. Only configurations above
        the ground are supported."""
        d = self.dh_d
        a = self.dh_a
        b = flange_pos[2] - d[0]

        if allclose(flange_pos[:2], 0):  # last link pointing up
            max_b = d[3] + a[1]
            if allclose(b, max_b):  # full length
                return [np.array([singularity_theta1, 0, 0])]
            if b > d[0]:
                arg1 = (a[1] ** 2 + b**2 - d[3] ** 2) / (2 * a[1] * b)
                arg2 = (a[1] ** 2 + d[3] ** 2 - b**2) / (2 * a[1] * d[3])
                if np.abs(arg1) > 1.0 or np.abs(arg2) > 1.0:
                    return []
                th2 = -np.arccos(arg1)
                th3 = np.pi - np.arccos(arg2)
                return [
                    np.array([singularity_theta1, th2, th3]),
                    np.array([singularity_theta1, -th2, -th3]),
                ]
            return []

        c = np.sqrt(b**2 + flange_pos[0] ** 2 + flange_pos[1] ** 2)
        if allclose(c, d[3] + a[1]):  # full length
            tmp = -np.pi / 2 + np.arcsin(b / c)
            return [
                np.array([np.arctan2(flange_pos[1], flange_pos[0]), tmp, 0]),
                np.array([np.arctan2(-flange_pos[1], -flange_pos[0]), -tmp, 0]),
            ]
        if c >= d[3] + a[1]:
            return []

        theta2_base = (
            np.pi / 2
            - np.arcsin(b / c)
            + np.arccos((a[1] ** 2 + c**2 - d[3] ** 2) / (2 * a[1] * c))
        )
        th2_term1 = np.arctan2(np.sin(theta2_base), np.cos(theta2_base))

        theta1_pos = np.arctan2(flange_pos[1], flange_pos[0])
        theta1_neg = np.arctan2(-flange_pos[1], -flange_pos[0])
        theta3_term1 = np.pi - np.arccos(
            (a[1] ** 2 + d[3] ** 2 - c**2) / (2 * a[1] * d[3])
        )

        th2_term2 = (
            -np.pi / 2
            + np.arcsin(b / c)
            + np.arccos((a[1] ** 2 + c**2 - d[3] ** 2) / (2 * a[1] * c))
        )

        return [
            np.array([theta1_pos, -th2_term1, theta3_term1]),
            np.array([theta1_neg, th2_term1, -theta3_term1]),
            np.array([theta1_pos, th2_term2, -theta3_term1]),
            np.array([theta1_neg, -th2_term2, theta3_term1]),
        ]

    def ik(self, pose: np.ndarray) -> list[np.ndarray]:
        """
        Analytical inverse kinematics. Like on the real robot, the solutions
        are returned in random order and are NOT filtered by joint limits.
        :param pose: 4x4 homogeneous matrix of the flange w.r.t. the robot base
        :return: list of joint configurations [rad] reaching the pose
        """
        # X=A01*A12*A23 * [0 0 0 1]' because A34*A45*A57==R34*R45*R56 is pure rotation
        flange_pos = pose @ np.array([0, 0, -self.dh_d[5], 1])
        sols_q_03 = self._ik_flange_pos(flange_pos)

        singularity_theta4 = 0

        sols = []
        for q_03 in sols_q_03:
            rot_03 = self.fk(q_03)[:3, :3]
            rot_36 = rot_03.T @ pose[:3, :3]

            # Euler Z - Y Z for joints 4, 5, 6
            P = rot_36
            if np.isclose(P[2][2], 1):  # np.cos(theta5) == 1
                sols.append(np.concatenate([
                    q_03,
                    [singularity_theta4, 0,
                     np.arctan2(P[1][0], P[0][0]) - singularity_theta4],
                ]))
            elif np.isclose(P[2][2], -1):  # np.cos(theta5) == -1
                sols.append(np.concatenate([
                    q_03,
                    [singularity_theta4, np.pi,
                     np.arctan2(P[1][0], -P[0][0]) + singularity_theta4],
                ]))
            else:  # non-degenerate
                theta5 = np.arccos(P[2][2])
                for sign in (1, -1):
                    s = np.sign(np.sin(sign * theta5))
                    sols.append(np.concatenate([
                        q_03,
                        [np.arctan2(P[1][2] * s, P[0][2] * s),
                         -sign * theta5,
                         np.arctan2(P[2][1] * s, -P[2][0] * s)],
                    ]))
        np.random.shuffle(sols)
        return sols
