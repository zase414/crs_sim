"""
Simulated CRS robot with the same interface as ``ctu_crs.CRSRobot``.

Code written against the real robot should run unchanged on the simulator:
``move_to_q`` is non-blocking and the robot moves in (scaled) real time, so
``get_q``, ``in_motion`` and ``wait_for_motion_stop`` behave like on hardware.
Kinematics, joint limits, IRC quantization and default speeds are the same
as in ``ctu_crs``.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike

from .camera import Camera, RobotBody
from .gripper import Gripper
from .kinematics import DHKinematics


@dataclass(frozen=True)
class _Motion:
    """Coordinated joint motion with a quintic time scaling."""
    q_start: np.ndarray
    q_goal: np.ndarray
    t_start: float
    duration: float

    def q_at(self, t: float) -> np.ndarray:
        if self.duration <= 0:
            return self.q_goal.copy()
        s = np.clip((t - self.t_start) / self.duration, 0.0, 1.0)
        s = s**3 * (10 - 15 * s + 6 * s**2)
        return self.q_start + s * (self.q_goal - self.q_start)

    def finished(self, t: float) -> bool:
        return t >= self.t_start + self.duration


# Peak velocity of the quintic profile relative to the average velocity
_QUINTIC_PEAK = 1.875

# How long the joint history is kept, longer than any camera lag [s]
_HISTORY_DURATION = 10.0


class CRSRobot(DHKinematics):
    """
    Simulated CRS robot.
    :param tty_dev: ignored, kept for compatibility with ctu_crs
    :param baudrate: ignored, kept for compatibility with ctu_crs
    :param urdf_path: URDF used for rendering
    :param visualize: open a MeshCat viewer and render the robot
    :param open_browser: open the MeshCat page in the web browser
    :param speedup: motion time scale, 1.0 = real speed, ``np.inf`` = instant
    :param q0: initial joint configuration, defaults to ``q_home``
    :param render_fps: rendering rate while moving
    :param camera: overhead camera returned by ``grab_image``: True for the
        default one (:class:`crs_sim.camera.Camera`), a ``Camera`` to
        configure it, False for none
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
        **crs_kwargs,
    ) -> None:
        super().__init__(link_lengths=[0.3052, 0.3048, 0.3302, 0.0762])
        self.urdf_path = Path(urdf_path)

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
        #: Image returned by grab_image() instead of the simulated camera,
        #: set it to feed your vision pipeline a recorded image
        self.camera_image: np.ndarray | None = None

        # Simulation state
        self.speedup = speedup
        self._max_velocity = np.zeros(6)
        self._set_speed_no_check(self._default_speed_irc256_per_ms)
        self._acceleration = self._default_acceleration_irc_per_ms.astype(float)
        q0 = self.q_home if q0 is None else np.asarray(q0, dtype=float)
        self._motion = _Motion(q0.copy(), q0.copy(), time.monotonic(), 0.0)
        # recent motions, so that q can be looked up in the past (camera lag)
        self._motions: deque[_Motion] = deque([self._motion])

        self.viz = None
        if visualize:
            from .visualizer import Visualizer
            self.viz = Visualizer(self.urdf_path, open_browser=open_browser)
            self.viz.display(q0)
            self._render_period = 1.0 / render_fps
            self._render_request = threading.Event()
            threading.Thread(target=self._render_loop, daemon=True).start()

        #: Simulated overhead camera (crs_sim.camera.Camera) or None
        self.camera: Camera | None = None
        if camera is not False:
            self.camera = Camera() if camera is True else camera
            self.camera.add(RobotBody(self))
            if self.viz is not None:
                self.camera.show(self.viz)

    # ------------------------------------------------------------------
    # Interface of ctu_crs.CRSRobot
    # ------------------------------------------------------------------
    def grab_image(self, timeout: int = 10000) -> np.ndarray:
        """
        Image of the simulated overhead camera (HxWx3 uint8, BGR), showing
        the scene ``camera.lag`` seconds ago. Returns ``self.camera_image``
        instead if it is set.
        """
        if self.camera_image is not None:
            return self.camera_image.copy()
        if self.camera is None:
            raise RuntimeError(
                "The robot was created with camera=False. Set "
                "robot.camera_image to an image to be returned by grab_image()."
            )
        return self.camera.grab()

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
        self._start_motion(q)

    def get_q(self) -> np.ndarray:
        """Get current joint configuration."""
        return self._motion.q_at(time.monotonic())

    def in_motion(self) -> bool:
        """Return whether the robot is in motion."""
        return not self._motion.finished(time.monotonic())

    def wait_for_motion_stop(self) -> None:
        """Wait until the robot stops moving."""
        remaining = (self._motion.t_start + self._motion.duration
                     - time.monotonic())
        if remaining > 0:
            time.sleep(remaining)

    def in_limits(self, q: ArrayLike) -> bool:
        """Return whether the given joint configuration is in joint limits."""
        return np.all(q >= self.q_min) and np.all(q <= self.q_max)

    # ------------------------------------------------------------------
    # Simulator only
    # ------------------------------------------------------------------
    def q_at(self, t: float) -> np.ndarray:
        """
        Joint configuration at time t (time.monotonic() clock), also in the
        recent past (the last 10 s are kept), e.g. at a camera exposure time.
        """
        motions = list(self._motions)
        for motion in reversed(motions):
            if motion.t_start <= t:
                return motion.q_at(t)
        return motions[0].q_start.copy()

    def add_object(self, obj, name: str | None = None) -> None:
        """
        Add an object (e.g. a crs_sim.rod.MovingRod) to the scene: the camera
        renders it and the viewer shows it.
        :param obj: object with ``surfaces_at(t)`` and optionally ``attach(viz, name)``
        :param name: name in the viewer, defaults to the class name
        """
        if self.camera is not None:
            self.camera.add(obj)
        if self.viz is not None and hasattr(obj, "attach"):
            obj.attach(self.viz, name or type(obj).__name__.lower())

    # ------------------------------------------------------------------
    # Simulation internals
    # ------------------------------------------------------------------
    def _set_speed_no_check(self, speed_irc256_ms: ArrayLike) -> None:
        irc_per_s = np.asarray(speed_irc256_ms, dtype=float) * 1000 / 256
        self._max_velocity = np.deg2rad(irc_per_s / np.abs(self._deg_to_irc))

    def _start_motion(self, q: ArrayLike) -> None:
        """Start a coordinated motion from the current configuration to q."""
        # The controller works in IRC, so the reached q is quantized
        q_goal = self._irc_to_joint_values(self._joint_values_to_irc(q))
        now = time.monotonic()
        q_start = self._motion.q_at(now)
        duration = (_QUINTIC_PEAK
                    * np.max(np.abs(q_goal - q_start) / self._max_velocity)
                    / self.speedup)
        self._motion = _Motion(q_start, q_goal, now, float(duration))
        self._motions.append(self._motion)
        while (len(self._motions) > 1
               and self._motions[1].t_start < now - _HISTORY_DURATION):
            self._motions.popleft()
        if self.viz is not None:
            self._render_request.set()

    def _render_loop(self) -> None:
        """Render the robot while it moves."""
        while True:
            self._render_request.wait()
            self._render_request.clear()
            while True:
                moving = self.in_motion()
                self.viz.display(self.get_q())
                if not moving:
                    break
                time.sleep(self._render_period)
