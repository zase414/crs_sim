"""
Contact checking between the robot and objects in the scene (puzzles, the
moving rod), so that touching the wire is detected like the referee would.

The robot is checked with its exact URDF visual meshes (the hoop included,
the URDF collision boxes are far too coarse), objects with their CAD meshes.
Mesh-mesh collision is done by coal (hpp-fcl), which ships with pinocchio.

:class:`crs_sim.CRSRobot` runs the check along every executed motion (see
``CRSRobot.contacts``), this module can also be used directly::

    checker = ContactChecker(robot.urdf_path)
    checker.add(puzzle)
    checker.check(q)          # -> [Contact(link, obstacle, point), ...]
    checker.distance(q)       # -> smallest robot-object distance [m]
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass(frozen=True, eq=False)
class Contact:
    """One touching robot link - object part pair."""
    link: str            #: robot link, e.g. "hoop"
    obstacle: str        #: object part, e.g. "puzzle_A/wire"
    point: np.ndarray    #: a contact point in the robot base frame [m]


@dataclass(frozen=True, eq=False)
class ContactEvent:
    """The robot started touching something."""
    time: float          #: time.monotonic() time
    q: np.ndarray        #: joint configuration
    contacts: list[Contact] = field(default_factory=list)

    def __str__(self) -> str:
        pairs = ", ".join(f"{c.link} <-> {c.obstacle}" for c in self.contacts)
        return f"contact {pairs} at q = {np.round(self.q, 3).tolist()}"


class ContactError(RuntimeError):
    """Raised by the robot (``on_contact="raise"``) after a contact."""
    def __init__(self, event: ContactEvent) -> None:
        super().__init__(str(event))
        self.event = event


def mesh_geometry(vertices: np.ndarray, faces: np.ndarray):
    """coal BVH geometry of a triangle mesh."""
    import coal

    bvh = coal.BVHModelOBBRSS()
    bvh.beginModel(len(faces), len(vertices))
    bvh.addVertices(np.asarray(vertices, dtype=float))
    bvh.addTriangles(np.asarray(faces, dtype=np.int64))
    bvh.endModel()
    return bvh


class ContactChecker:
    """
    Robot vs. objects collision checker.

    Objects provide ``collision_geometries_at(t)`` returning
    ``[(name, coal geometry, 4x4 pose), ...]`` (see :class:`crs_sim.puzzles.Puzzle`,
    :class:`crs_sim.rod.MovingRod`).
    :param urdf_path: robot URDF, its visual meshes are used
    :param margin: report contacts already when closer than this [m]
    """
    def __init__(self, urdf_path: str | Path, margin: float = 0.0) -> None:
        import coal
        import pinocchio as pin

        self._coal, self._pin = coal, pin
        urdf_path = Path(urdf_path)
        self._model, _, self._geom = pin.buildModelsFromUrdf(
            str(urdf_path), package_dirs=[str(urdf_path.parent)])
        self._data = self._model.createData()
        self._geom_data = pin.GeometryData(self._geom)
        # robot link of each visual geometry
        self.link_names = [self._model.frames[g.parentFrame].name
                           for g in self._geom.geometryObjects]
        self.margin = margin
        self.objects: list = []
        self._lock = threading.Lock()

    def add(self, obj) -> None:
        """Add an object with ``collision_geometries_at(t)``."""
        with self._lock:
            if obj not in self.objects:
                self.objects.append(obj)

    def remove(self, obj) -> None:
        with self._lock:
            if obj in self.objects:
                self.objects.remove(obj)

    def _obstacles(self, t: float):
        with self._lock:
            objects = list(self.objects)
        return [(name, geom, self._transform(pose)) for obj in objects
                for name, geom, pose in obj.collision_geometries_at(t)]

    def _transform(self, pose: np.ndarray):
        pose = np.asarray(pose, dtype=float)
        return self._coal.Transform3s(pose[:3, :3], pose[:3, 3])

    def _links(self, q: np.ndarray):
        pin = self._pin
        pin.forwardKinematics(self._model, self._data, np.asarray(q, dtype=float))
        pin.updateGeometryPlacements(self._model, self._data, self._geom,
                                     self._geom_data)
        return [(self.link_names[k], g.geometry,
                 self._transform(self._geom_data.oMg[k].homogeneous))
                for k, g in enumerate(self._geom.geometryObjects)]

    def check(self, q: np.ndarray, t: float | None = None) -> list[Contact]:
        """
        Robot links touching an object at joint configuration q.
        :param t: time.monotonic() time the (moving) objects are posed at
        """
        coal = self._coal
        t = time.monotonic() if t is None else t
        obstacles = self._obstacles(t)
        if not obstacles:
            return []
        request = coal.CollisionRequest(coal.CollisionRequestFlag.CONTACT, 1)
        request.security_margin = self.margin
        contacts = []
        for link, lgeom, ltf in self._links(q):
            for name, ogeom, otf in obstacles:
                result = coal.CollisionResult()
                if coal.collide(lgeom, ltf, ogeom, otf, request, result):
                    c = result.getContact(0)
                    point = np.array(c.pos) if self.margin == 0 else (
                        (np.array(c.nearest_points[0])
                         + np.array(c.nearest_points[1])) / 2)
                    contacts.append(Contact(link, name, point))
        return contacts

    def distance(self, q: np.ndarray, t: float | None = None,
                 links: list[str] | None = None) -> float:
        """
        Smallest distance between the robot and the objects [m] (negative
        when they penetrate, inf without objects).
        :param links: only these robot links, e.g. ["hoop"]
        """
        coal = self._coal
        t = time.monotonic() if t is None else t
        best = np.inf
        obstacles = self._obstacles(t)
        for link, lgeom, ltf in self._links(q) if obstacles else []:
            if links is not None and link not in links:
                continue
            for _, ogeom, otf in obstacles:
                result = coal.DistanceResult()
                best = min(best, coal.distance(lgeom, ltf, ogeom, otf,
                                               coal.DistanceRequest(), result))
        return best
