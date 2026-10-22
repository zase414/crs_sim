"""
Simulated CRS97 robot, drop-in replacement of ``ctu_crs.CRS97``.
"""
from __future__ import annotations

from pathlib import Path

from .crs_robot import CRSRobot

URDF_PATH = Path(__file__).parent / "urdf" / "CRS_97.urdf"

# Same values as ctu_crs/params_crs97.yaml
CRS97_PARAMS = {
    "hh_irc": [-182500, 252, -63625, 99200, 14300, -98150],
    "hh_sequence": ["B", "AC", "DEF"],
    "lower_bound_irc": [-370000, -100000, 65000, 202000, 72000, 4000],
    "upper_bound_irc": [10000, 100000, -190000, -5000, -44000, -199500],
    "default_speed_irc256_per_ms": [7680, 2048, 5120, 7680, 7680, 14080],
    "default_acceleration_irc_per_ms": [19, 5, 13, 19, 128, 154],
    "gripper": {"bounds": [1001, 48], "axis": "H"},
    "camera_name": "camera-crs97",
}


class CRS97(CRSRobot):
    """
    Simulated CRS97. Accepts the same arguments as ``ctu_crs.CRS97`` plus the
    keyword-only simulator options of :class:`CRSRobot` (``visualize``,
    ``open_browser``, ``speedup``, ``q0``, ``render_fps``, ``camera``).
    """
    def __init__(self, tty_dev: str | None = "/dev/mars", baudrate: int = 19200,
                 **sim_kwargs) -> None:
        sim_kwargs.setdefault("urdf_path", URDF_PATH)
        super().__init__(tty_dev, baudrate, **sim_kwargs, **CRS97_PARAMS)
