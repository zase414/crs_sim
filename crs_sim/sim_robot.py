"""
Simulation machinery shared by the simulated robots (:class:`crs_sim.CRSRobot`,
:class:`crs_sim.Rv6s`): joint motions in (scaled) real time, the MeshCat
viewer, the overhead camera and contact checking against added objects.
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
from .contact import ContactError, ContactEvent
from .transforms import translation


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

# Conservative distance of any robot point from any joint axis, bounds how
# far a point moves for a given joint step [m]
_REACH = 0.8


class SimRobot:
    """
    Base of the simulated robots. A subclass sets ``q_home`` and
    ``_max_velocity`` (rad/s per joint, the motion duration follows from it)
    before calling this ``__init__``.
    :param urdf_path: URDF used for rendering and contact checking
    :param visualize: open a MeshCat viewer and render the robot
    :param open_browser: open the MeshCat page in the web browser
    :param speedup: motion time scale, 1.0 = real speed, ``np.inf`` = instant
    :param q0: initial joint configuration, defaults to ``q_home``
    :param render_fps: rendering rate while moving
    :param camera: overhead camera returned by ``grab_image``: True for the
        default one (:class:`crs_sim.camera.Camera`), a ``Camera`` to
        configure it, False for none
    :param on_contact: what happens when the robot touches an added object
        (see ``contacts``): "warn" prints it and marks the spot in the viewer,
        "raise" also raises ContactError from the next move_to_q /
        wait_for_motion_stop, "ignore" only records it
    :param contact_resolution: largest distance any robot point moves between
        two contact checks along a motion [m]
    :param contact_margin: count it as contact already when the robot is
        closer than this to an object [m]
    """
    #: Hoop center in the flange frame returned by ``fk``, the wire passes
    #: through the hoop along this frame's z axis. The universal hoop sits on
    #: the same adapter stack on every robot (crs_sim.puzzles.HOOP_IN_FLANGE).
    hoop_in_flange = translation([-0.135, 0.0, 0.012])

    def __init__(
        self,
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
    ) -> None:
        self.urdf_path = Path(urdf_path)
        #: Image returned by grab_image() instead of the simulated camera,
        #: set it to feed your vision pipeline a recorded image
        self.camera_image: np.ndarray | None = None

        self.speedup = speedup
        q0 = self.q_home if q0 is None else np.asarray(q0, dtype=float)
        self._motion = _Motion(q0.copy(), q0.copy(), time.monotonic(), 0.0)
        # recent motions, so that q can be looked up in the past (camera lag)
        self._motions: deque[_Motion] = deque([self._motion])

        if on_contact not in ("warn", "raise", "ignore"):
            raise ValueError(f"on_contact must be warn, raise or ignore, not {on_contact!r}")
        self.on_contact = on_contact
        self.contact_resolution = contact_resolution
        self.contact_margin = contact_margin
        #: crs_sim.contact.ContactChecker, created when the first object with
        #: collision geometry is added
        self.contact_checker = None
        self._contact_events: list[ContactEvent] = []
        self._contacts_raised = 0
        self._in_contact = False
        # motions not fully checked yet, the first one is being checked
        self._contact_queue: deque[_Motion] = deque()
        self._contact_cond = threading.Condition()
        self._contact_idle = True
        self._object_names: dict[int, str] = {}
        self._monitors: list = []

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
        Add an object (e.g. a crs_sim.puzzles.Puzzle or a
        crs_sim.rod.MovingRod) to the scene: the camera renders it, the
        viewer shows it and touching it is detected (see ``contacts``).
        :param obj: object with ``surfaces_at(t)`` and optionally
            ``attach(viz, name)`` and ``collision_geometries_at(t)``
        :param name: name in the viewer, the object's default if None
        """
        if self.camera is not None:
            self.camera.add(obj)
        if self.viz is not None and hasattr(obj, "attach"):
            if name is None:
                obj.attach(self.viz)
            else:
                obj.attach(self.viz, name)
        if hasattr(obj, "collision_geometries_at"):
            self._start_contact_checking()
            self.contact_checker.add(obj)

    def remove_object(self, obj) -> None:
        """Remove an object added with add_object from the scene."""
        if self.camera is not None and obj in self.camera.objects:
            self.camera.remove(obj)
        if self.contact_checker is not None:
            self.contact_checker.remove(obj)
        if hasattr(obj, "detach"):
            obj.detach()

    @property
    def contacts(self) -> list[ContactEvent]:
        """
        Every time the robot started touching an added object, in the order
        it happened (a contact lasting over several checks is one event).
        Every executed motion is checked, finely enough that no robot point
        moves more than ``contact_resolution`` between two checks.
        """
        with self._contact_cond:
            return list(self._contact_events)

    def clear_contacts(self) -> None:
        """Forget the recorded contacts (e.g. before the next attempt)."""
        with self._contact_cond:
            self._contact_events.clear()
            self._contacts_raised = 0
        if self.viz is not None:
            self.viz.delete("contacts")

    def add_monitor(self, callback) -> None:
        """
        Call ``callback(q, t)`` for every sample of the executed path, the
        same dense samples the contact check uses (in order, from the
        contact checking thread). Contact checking starts if it is not running.
        """
        self._start_contact_checking()
        self._monitors.append(callback)

    def settle(self) -> None:
        """Wait until the robot stops and every executed motion is checked
        for contacts (never raises ContactError)."""
        remaining = (self._motion.t_start + self._motion.duration
                     - time.monotonic())
        if remaining > 0:
            time.sleep(remaining)
        with self._contact_cond:
            self._contact_cond.wait_for(lambda: self._contact_idle)

    def in_contact(self, q: ArrayLike | None = None) -> bool:
        """Whether the robot touches an added object at q (now if None)."""
        if self.contact_checker is None:
            return False
        q = self._current_q() if q is None else np.asarray(q, dtype=float)
        return bool(self.contact_checker.check(q))

    # ------------------------------------------------------------------
    # Simulation internals
    # ------------------------------------------------------------------
    def _current_q(self) -> np.ndarray:
        return self._motion.q_at(time.monotonic())

    def _quantize(self, q: ArrayLike) -> np.ndarray:
        """The configuration the controller really reaches when sent to q."""
        return np.asarray(q, dtype=float).copy()

    def _start_motion(self, q: ArrayLike) -> None:
        """Start a coordinated motion from the current configuration to q."""
        q_goal = self._quantize(q)
        now = time.monotonic()
        q_start = self._motion.q_at(now)
        duration = (_QUINTIC_PEAK
                    * np.max(np.abs(q_goal - q_start) / self._max_velocity)
                    / self.speedup)
        self._motion = _Motion(q_start, q_goal, now, float(duration))
        self._motions.append(self._motion)
        if self.contact_checker is not None:
            with self._contact_cond:
                self._contact_queue.append(self._motion)
                self._contact_idle = False
                self._contact_cond.notify_all()
        while (len(self._motions) > 1
               and self._motions[1].t_start < now - _HISTORY_DURATION):
            self._motions.popleft()
        if self.viz is not None:
            self._render_request.set()

    def _wait_for_motion(self, timeout: float | None = None) -> bool:
        """
        Sleep until the motion ends (at most timeout seconds), let the
        contact check catch up and raise a contact (on_contact="raise").
        Return whether the robot stopped.
        """
        remaining = (self._motion.t_start + self._motion.duration
                     - time.monotonic())
        if timeout is not None and remaining > timeout:
            time.sleep(max(0.0, timeout))
            self._raise_contacts()
            return False
        # the contact check of the motion may still be catching up
        self.settle()
        self._raise_contacts()
        return True

    def _render_loop(self) -> None:
        """Render the robot while it moves."""
        while True:
            self._render_request.wait()
            self._render_request.clear()
            while True:
                moving = not self._motion.finished(time.monotonic())
                self.viz.display(self._current_q())
                if not moving:
                    break
                time.sleep(self._render_period)

    # ------------------------------------------------------------------
    # Contact checking
    # ------------------------------------------------------------------
    def _start_contact_checking(self) -> None:
        """Create the checker and check motions from now on."""
        if self.contact_checker is not None:
            return
        from .contact import ContactChecker

        self.contact_checker = ContactChecker(self.urdf_path,
                                              margin=self.contact_margin)
        with self._contact_cond:
            now = time.monotonic()
            q = self._motion.q_at(now)
            # check from the current configuration on
            self._contact_queue.append(
                _Motion(q, self._motion.q_goal, now,
                        max(0.0, self._motion.t_start + self._motion.duration
                            - now))
                if not self._motion.finished(now) else _Motion(q, q, now, 0.0))
            self._contact_idle = False
        threading.Thread(target=self._contact_loop, daemon=True).start()

    def _contact_samples(self, motion: _Motion) -> int:
        """Number of check intervals along a motion."""
        dq = np.sum(np.abs(motion.q_goal - motion.q_start))
        return int(np.ceil(_QUINTIC_PEAK * _REACH * dq / self.contact_resolution))

    def _contact_loop(self) -> None:
        """
        Check the executed part of every motion, in order, keeping up with
        real time: a sample is checked once the robot has passed it, so a
        motion replaced by the next move_to_q is only checked up to where
        the robot really was.
        """
        k = 0
        while True:
            with self._contact_cond:
                motion = self._contact_queue[0]
                nxt = self._contact_queue[1] if len(self._contact_queue) > 1 else None
            n = self._contact_samples(motion)
            t_end = np.inf if nxt is None else nxt.t_start
            now = time.monotonic()
            while k <= n:
                t = motion.t_start + (motion.duration * k / n if n else 0.0)
                if t > now or (k > 0 and t > t_end):
                    break
                self._check_contact(motion.q_at(t), t)
                k += 1
            done = k > n or (nxt is not None and t > t_end)
            with self._contact_cond:
                if done and len(self._contact_queue) > 1:
                    self._contact_queue.popleft()
                    k = 0
                    continue
                if done:
                    self._contact_idle = True
                    self._contact_cond.notify_all()
                    self._contact_cond.wait_for(
                        lambda: len(self._contact_queue) > 1)
                    self._contact_queue.popleft()
                    k = 0
                else:
                    # caught up with the robot, wait for it to move on
                    self._contact_cond.wait(min(0.005, t - now))

    def _check_contact(self, q: np.ndarray, t: float) -> None:
        for callback in self._monitors:
            callback(q, t)
        contacts = self.contact_checker.check(q, t)
        if contacts and not self._in_contact:
            event = ContactEvent(t, q, contacts)
            with self._contact_cond:
                self._contact_events.append(event)
            if self.on_contact != "ignore":
                print(f"CONTACT: {event}")
                if self.viz is not None:
                    points = [c.point for e in self.contacts for c in e.contacts]
                    self.viz.show_points("contacts", np.array(points),
                                         color=0xff0000, radius=0.004)
        self._in_contact = bool(contacts)

    def _raise_contacts(self) -> None:
        """Raise ContactError for a contact not raised yet (on_contact="raise")."""
        if self.on_contact != "raise":
            return
        with self._contact_cond:
            if self._contacts_raised < len(self._contact_events):
                event = self._contact_events[self._contacts_raised]
                self._contacts_raised = len(self._contact_events)
                raise ContactError(event)
