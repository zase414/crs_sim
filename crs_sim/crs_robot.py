"""
Simulated CRS robot with the same interface as ``ctu_crs.CRSRobot``.

Code written against the real robot should run unchanged on the simulator:
``move_to_q`` is non-blocking and the robot moves in (scaled) real time, so
``get_q``, ``in_motion`` and ``wait_for_motion_stop`` behave like on hardware.
Kinematics, joint limits, IRC quantization and default speeds are the same
as in ``ctu_crs``.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike

from .camera import Camera
from .gripper import Gripper
from .kinematics import DHKinematics
from .sim_robot import SimRobot


class CRSRobot(DHKinematics, SimRobot):
    """
    Simulated CRS robot.
    :param tty_dev: ignored, kept for compatibility with ctu_crs
    :param baudrate: ignored, kept for compatibility with ctu_crs
    :param urdf_path, visualize, open_browser, speedup, q0, render_fps,
        camera, on_contact, contact_resolution, contact_margin: simulator
        options, see :class:`crs_sim.sim_robot.SimRobot`
    :param crs_kwargs: robot parameters (see ``crs97.CRS97_PARAMS``)
    """
    def __init__(
        self,
        tty_dev: str | None = "/dev/mars",
        baudrate: int = 19200,
        *,
        urdf_path: str | Path,
        visualize: bool = True,
        open_browser: bool = True,
        speedup: float = 1.0,
        q0: ArrayLike | None = None,
        render_fps: float = 30.0,
        camera: bool | Camera = True,
        on_contact: str = "warn",
        contact_resolution: float = 0.001,
        contact_margin: float = 0.0,
        **crs_kwargs,
    ) -> None:
        DHKinematics.__init__(self, link_lengths=[0.3052, 0.3048, 0.3302, 0.0762])

        # conversion IRC to radians
        irc = np.array([1000, 1000, 1000, 500, 500, 500])  # IRC per rotation of motor.
        gearing = np.array([100, 100, 100, 101, 100, 101])
        direction = np.array([1, 1, -1, -1, -1, -1])
        self._deg_to_irc = irc * direction * gearing * 4 / 360

        self._motors_ids = "ABCDEF"

        self._hh_rad = np.array([0, 0, 0, 0, 0, 0])
        self._hh_irc = np.array(crs_kwargs["hh_irc"], dtype=int)
        self._hh_sequence = crs_kwargs.get("hh_sequence", [])

        self.q_min = self._irc_to_joint_values(crs_kwargs["lower_bound_irc"])
        self.q_max = self._irc_to_joint_values(crs_kwargs["upper_bound_irc"])
        self.q_home = np.deg2rad([0, 0, -45, 0, -45, 0])

        self._default_speed_irc256_per_ms = np.array(
            crs_kwargs["default_speed_irc256_per_ms"], dtype=int
        )
        self._min_speed_irc256_per_ms = np.rint(self._default_speed_irc256_per_ms / 5)
        self._max_speed_irc256_per_ms = np.rint(self._default_speed_irc256_per_ms * 2)

        self._default_acceleration_irc_per_ms = np.array(
            crs_kwargs["default_acceleration_irc_per_ms"], dtype=int
        )
        self._min_acceleration_irc_per_ms = np.rint(
            self._default_acceleration_irc_per_ms / 5
        )
        self._max_acceleration_irc_per_ms = np.rint(
            self._default_acceleration_irc_per_ms * 2
        )

        self.gripper = Gripper(**crs_kwargs["gripper"])

        self._initialized = False
        self._camera_name: str = crs_kwargs["camera_name"]

        self._max_velocity = np.zeros(6)
        self._set_speed_no_check(self._default_speed_irc256_per_ms)
        self._acceleration = self._default_acceleration_irc_per_ms.astype(float)
        SimRobot.__init__(
            self, urdf_path=urdf_path, visualize=visualize,
            open_browser=open_browser, speedup=speedup, q0=q0,
            render_fps=render_fps, camera=camera, on_contact=on_contact,
            contact_resolution=contact_resolution, contact_margin=contact_margin)

    # ------------------------------------------------------------------
    # Interface of ctu_crs.CRSRobot
    # ------------------------------------------------------------------
    def release(self) -> None:
        """Release errors and reset control unit (no-op)."""

    def reset_motors(self) -> None:
        """Reset motors of robot (no-op)."""

    def close(self) -> None:
        """Close connection to the robot (no-op)."""

    def initialize(self, home: bool = True) -> None:
        """
        Initialize the robot: reset speed and acceleration to defaults and
        perform hard home and soft home, if @param home is True.
        """
        self.set_speed(self._default_speed_irc256_per_ms)
        self.set_acceleration(self._default_acceleration_irc_per_ms)
        self.gripper.initialize()
        if home:
            self.hard_home()
            self.soft_home()
        self._initialized = True

    def _joint_values_to_irc(self, joint_values: ArrayLike) -> np.ndarray:
        """Convert joint values [rad] to IRC."""
        j = np.asarray(joint_values)
        assert j.shape == (len(self._motors_ids),), "Incorrect number of joints."
        irc = np.rad2deg(j + self._hh_rad) * self._deg_to_irc + self._hh_irc
        return np.rint(irc)

    def _irc_to_joint_values(self, irc: ArrayLike) -> np.ndarray:
        """Convert IRC to joint values [rad]."""
        irc = np.asarray(irc)
        assert irc.shape == (len(self._motors_ids),), "Incorrect number of joints."
        return np.deg2rad((irc - self._hh_irc) / self._deg_to_irc) + self._hh_rad

    def set_speed(self, speed_irc256_ms: ArrayLike) -> None:
        """Set speed for each motor in IRC*256/msec."""
        assert len(speed_irc256_ms) == len(self._motors_ids)
        self._set_speed_no_check(speed_irc256_ms)

    def set_speed_relative(self, fraction: float) -> None:
        """Set speed for each motor in fraction (0-1) of maximum speed."""
        assert 0 <= fraction <= 1, "Fraction must be in [0,1]."
        s = self._min_speed_irc256_per_ms + fraction * (
            self._max_speed_irc256_per_ms - self._min_speed_irc256_per_ms
        )
        self.set_speed(s)

    def set_acceleration(self, acceleration_irc_ms: ArrayLike) -> None:
        """Set acceleration for each motor in IRC/msec. Stored only, the
        simulator uses a smooth velocity profile instead."""
        assert len(acceleration_irc_ms) == len(self._motors_ids)
        self._acceleration = np.asarray(acceleration_irc_ms, dtype=float)

    def set_acceleration_relative(self, fraction: float) -> None:
        """Set acceleration for each motor in fraction (0-1) of maximum acceleration."""
        assert 0 <= fraction <= 1, "Fraction must be in [0,1]."
        a = self._min_acceleration_irc_per_ms + fraction * (
            self._max_acceleration_irc_per_ms - self._min_acceleration_irc_per_ms
        )
        self.set_acceleration(a)

    def hard_home(self) -> None:
        """Simulate hard home: joints are driven to their home switches
        (q = 0) block by block following the hard home sequence."""
        self.set_speed(self._default_speed_irc256_per_ms)
        self.set_acceleration(self._default_acceleration_irc_per_ms)
        if len(self._hh_sequence) == 0:
            raise ValueError("The hard home sequence is not defined for this robot.")
        for blk in self._hh_sequence:
            q = self.get_q()
            for axis in blk:
                q[self._motors_ids.index(axis)] = self._hh_rad[
                    self._motors_ids.index(axis)]
            self._start_motion(q)
            self.wait_for_motion_stop()

    def soft_home(self) -> None:
        """Move robot to the home position using coordinated movement."""
        self._start_motion(self.q_home)
        self.wait_for_motion_stop()

    def move_to_q(self, q: ArrayLike) -> None:
        """Move robot to the given joint configuration [rad] using coordinated
        movement. Non-blocking, use wait_for_motion_stop() to wait."""
        assert self._initialized, "You need to initialize the robot before moving it."
        assert self.in_limits(q), "Joint limits violated."
        self._raise_contacts()
        self._start_motion(q)

    def get_q(self) -> np.ndarray:
        """Get current joint configuration."""
        return self._current_q()

    def in_motion(self) -> bool:
        """Return whether the robot is in motion."""
        return not self._motion.finished(time.monotonic())

    def wait_for_motion_stop(self) -> None:
        """Wait until the robot stops moving."""
        self._wait_for_motion()

    def in_limits(self, q: ArrayLike) -> bool:
        """Return whether the given joint configuration is in joint limits."""
        return np.all(q >= self.q_min) and np.all(q <= self.q_max)

    # ------------------------------------------------------------------
    # Simulation internals
    # ------------------------------------------------------------------
    def _set_speed_no_check(self, speed_irc256_ms: ArrayLike) -> None:
        irc_per_s = np.asarray(speed_irc256_ms, dtype=float) * 1000 / 256
        self._max_velocity = np.deg2rad(irc_per_s / np.abs(self._deg_to_irc))

    def _quantize(self, q: ArrayLike) -> np.ndarray:
        """The controller works in IRC, so the reached q is quantized."""
        return self._irc_to_joint_values(self._joint_values_to_irc(q))
