"""
Standalone simulator of the CRS97 and the Mitsubishi RV-6S robotic arms with
the interfaces of ``ctu_crs`` and ``ctu_mitsubishi``.

    from crs_sim import CRS97      # instead of: from ctu_crs import CRS97
    from crs_sim import Rv6s       # instead of: from ctu_mitsubishi import Rv6s
"""
from .camera import Camera
from .contact import ContactChecker, ContactError
from .crs97 import CRS97, CRS97_PARAMS, URDF_PATH
from .crs_robot import CRSRobot
from .puzzles import Puzzle
from .rod import MovingRod
from .rv6s import URDF_PATH_RV6S, Rv6s, Rv6sGripper
from .sim_robot import SimRobot
from .visualizer import Visualizer, load_mesh

__all__ = ["CRS97", "CRS97_PARAMS", "CRSRobot", "Camera", "ContactChecker",
           "ContactError", "MovingRod", "Puzzle", "Rv6s", "Rv6sGripper",
           "SimRobot", "URDF_PATH", "URDF_PATH_RV6S", "Visualizer", "load_mesh"]
