"""
A moving rod to slide the hoop onto. Its pose is a function of time, it is
rendered by the simulated camera (lagged, like everything the camera sees)
and animated in the MeshCat viewer.

    from crs_sim.rod import MovingRod, oscillating
    from crs_sim.transforms import translation

    rod = MovingRod(oscillating(translation([0.45, 0, 0.05]), [0, 0.1, 0], 4.0))
    robot.add_object(rod)
    rod.pose()          # ground truth now, rod.top() its upper end

Rod frame: origin at the center of the bottom face, z along the rod axis.
"""
from __future__ import annotations

import threading
import time
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike

from .camera import Surface, color_rgb, cylinder_surface
from .transforms import translation

Trajectory = Callable[[float], np.ndarray]


def static(pose: np.ndarray) -> Trajectory:
    """Rod standing still at pose."""
    pose = np.array(pose, dtype=float)
    return lambda t: pose


def linear(start: np.ndarray, velocity: ArrayLike) -> Trajectory:
    """
    Rod moving with a constant velocity.
    :param start: 4x4 pose at t = 0
    :param velocity: [vx, vy, vz] in the base frame [m/s]
    """
    start = np.array(start, dtype=float)
    velocity = np.asarray(velocity, dtype=float)
    return lambda t: translation(velocity * t) @ start


def oscillating(center: np.ndarray, amplitude: ArrayLike,
                period: float) -> Trajectory:
    """
    Rod moving back and forth along a line, sinusoidally.
    :param center: 4x4 pose in the middle of the motion
    :param amplitude: [x, y, z] largest displacement from center [m]
    :param period: duration of one back-and-forth [s]
    """
    center = np.array(center, dtype=float)
    amplitude = np.asarray(amplitude, dtype=float)
    return lambda t: translation(amplitude * np.sin(2 * np.pi * t / period)) @ center


def circular(center: np.ndarray, radius: float, period: float) -> Trajectory:
    """
    Rod moving along a horizontal circle (keeping its orientation).
    :param center: 4x4 pose at the circle center
    :param radius: [m]
    :param period: duration of one revolution [s]
    """
    center = np.array(center, dtype=float)

    def trajectory(t: float) -> np.ndarray:
        a = 2 * np.pi * t / period
        return translation([radius * np.cos(a), radius * np.sin(a), 0]) @ center
    return trajectory


class MovingRod:
    """
    Cylindrical rod moving along a trajectory.
    :param trajectory: function of time since start [s] -> 4x4 pose of the rod
        frame in the robot base frame, or a fixed 4x4 pose
    :param radius: [m]
    :param length: [m]
    :param color: RGB hex color (bright colors are easy to segment)
    :param spacing: surface sampling distance for the camera [m]
    """
    def __init__(self, trajectory: Trajectory | np.ndarray, radius: float = 0.01,
                 length: float = 0.25, color: int = 0xe03c1e,
                 spacing: float = 0.0015) -> None:
        if not callable(trajectory):
            trajectory = static(trajectory)
        self.trajectory = trajectory
        self.radius = radius
        self.length = length
        self.color = color
        self.surface: Surface = cylinder_surface(radius, length, spacing,
                                                 color_rgb(color))
        #: Start of the trajectory (time.monotonic() clock)
        self.t0 = time.monotonic()
        self._viz = None
        self._name = None

    def restart(self) -> None:
        """Start the trajectory again from t = 0."""
        self.t0 = time.monotonic()

    def pose(self, t: float | None = None) -> np.ndarray:
        """
        Ground truth pose of the rod frame (bottom center, z along the rod).
        :param t: time.monotonic() time, now if None
        """
        t = time.monotonic() if t is None else t
        return np.asarray(self.trajectory(t - self.t0), dtype=float)

    def top(self, t: float | None = None) -> np.ndarray:
        """Ground truth position of the center of the rod's upper end."""
        return (self.pose(t) @ [0, 0, self.length, 1])[:3]

    def surfaces_at(self, t: float) -> list[tuple[Surface, np.ndarray]]:
        return [(self.surface, self.pose(t))]

    def attach(self, viz, name: str = "rod", fps: float = 30.0) -> None:
        """
        Draw the rod in the viewer and keep moving it there.
        :param viz: crs_sim.Visualizer
        :param name: object name in the viewer
        """
        first = self._viz is None
        self._viz, self._name = viz, name
        viz.show_cylinder(name, self.radius, self.length, self.pose(),
                          color=self.color)
        if first:
            threading.Thread(target=self._animate, args=(1.0 / fps,),
                             daemon=True).start()

    def _animate(self, period: float) -> None:
        while True:
            self._viz.set_pose(self._name, self.pose())
            time.sleep(period)
