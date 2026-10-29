"""
The ROB1 hoop-and-wire puzzles (A - E) as scene objects: CAD models, ArUco
markers of the base plate and contact checking, plus the wire center line of
puzzle A as an example of a path.

Each puzzle is a bent 10x10 mm wire plugged into a base plate. The task is to
thread the hoop mounted on the robot over the wire from its free end down to
the base and back without touching it. The CAD models (STEP) come from
https://github.com/pilcsimo/ROB1-2025-hw (GPL-3.0) and are downloaded on demand
by :func:`download_puzzles`, they are not part of this package.

    puzzle = Puzzle("B", random_pose(seed))
    robot.add_object(puzzle)     # viewer + camera + contact checking

A puzzle can turn on a turntable: ``Puzzle("B", pose, rotation_period=20,
rotation_center=[0.05, 0])`` rotates it about the vertical (maze z) axis
through a point that need not be the puzzle center.

Frames: a maze frame has its origin where the wire stands on the base plate
top (the plate center), z up. The base plate frame is the center of the
plate's bottom face, ``BASE_PLATE_THICKNESS`` below. Paths are hoop-center
poses whose z axis is tangent to the wire.
"""
from __future__ import annotations

import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike

from .transforms import densify, pose, rx, rz, translation

#: Hoop center in the flange frame returned by ``CRS97.fk`` (and
#: ``Rv6s.fk``, the same adapter stack, see ``robot.hoop_in_flange``). The
#: wire passes through the hoop along this frame's z axis.
HOOP_IN_FLANGE = translation([-0.135, 0.0, 0.012])
#: Inner radius of the hoop opening [m]
HOOP_INNER_RADIUS = 0.03

#: Base plate thickness, the puzzle stands on its top face [m]
BASE_PLATE_THICKNESS = 0.005
#: The black marker cells stand out of the plate top by [m]
MARKER_HEIGHT = 0.0006
#: ArUco markers on the base plate (DICT_4X4_50): side length [m] and
#: id -> center [x, y] in the maze frame
MARKER_SIZE = 0.04
MARKER_CENTERS = {1: (-0.0375, 0.0375), 2: (0.0375, -0.0375)}

PUZZLES_REPO_URL = "https://github.com/pilcsimo/ROB1-2025-hw"
_RAW_URL = "https://raw.githubusercontent.com/pilcsimo/ROB1-2025-hw/main/"
PUZZLE_FILES = {
    **{name: f"puzzles/{name}.step" for name in "ABCDE"},
    "base": "bases/base 01-02.step",
    "hoop": "robot_ee/Hoop.step",
}


def default_cache_dir() -> Path:
    """Directory the CAD files are downloaded to."""
    return Path.home() / ".cache" / "crs_sim" / "ROB1-2025-hw"


def download_puzzles(cache_dir: str | Path | None = None,
                     names: list[str] | None = None) -> dict[str, Path]:
    """
    Download the STEP models of the puzzles (skips files already present).
    :param cache_dir: target directory, :func:`default_cache_dir` if None
    :param names: subset of PUZZLE_FILES keys, all if None
    :return: name -> local path
    """
    cache_dir = Path(cache_dir) if cache_dir is not None else default_cache_dir()
    paths = {}
    for name in names or PUZZLE_FILES:
        rel = PUZZLE_FILES[name]
        path = cache_dir / rel
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            url = _RAW_URL + urllib.parse.quote(rel)
            print(f"Downloading {url}")
            tmp = path.with_suffix(".part")
            urllib.request.urlretrieve(url, tmp)
            tmp.rename(path)
        paths[name] = path
    return paths


# ----------------------------------------------------------------------
# Wire center lines
# ----------------------------------------------------------------------
# Segments, applied in order:
#   ("start", xyz, rotation)  first hoop pose (maze frame)
#   ("line", dxyz, n)         move by dxyz in the current hoop frame, n poses
# The current hoop frame has z pointing back along the wire, so "up the wire"
# is -z. Only puzzle A is provided, as an example of the format.
MAZES: dict[str, list[tuple]] = {
    "A": [
        ("start", [0, 0, 0.01], rz(np.pi) @ rx(np.pi)),
        ("line", [0, 0, -0.10], 2),
        ("line", [0, 0, -0.11], 2),
    ],
}


def maze_path(name: str) -> np.ndarray:
    """
    Hoop-center poses along the wire, from the base plate to past the free
    end, in the maze frame.
    :param name: key of MAZES ("A")
    :return: Nx4x4 poses
    """
    poses: list[np.ndarray] = []
    for seg in MAZES[name]:
        kind = seg[0]
        if kind == "start":
            start = translation(seg[1])
            start[:3, :3] = seg[2][:3, :3]
            poses.append(start)
        elif kind == "line":
            step = translation(np.asarray(seg[1], dtype=float) / seg[2])
            for _ in range(seg[2]):
                poses.append(poses[-1] @ step)
        else:
            raise ValueError(f"Unknown segment {kind}")
    # drop repeated poses
    keep = [0] + [i for i in range(1, len(poses))
                  if not np.allclose(poses[i], poses[i - 1])]
    return np.array(poses)[keep]




# ----------------------------------------------------------------------
# Puzzle in the scene
# ----------------------------------------------------------------------
class Puzzle:
    """
    A puzzle (wire A - E plugged into the base plate with the two ArUco
    markers) standing in the workspace. Add it to the robot to get it drawn
    in the viewer, rendered by the camera and checked for contacts::

        puzzle = Puzzle("C", pose([0.45, 0.05, 0.0], yaw=0.3))
        robot.add_object(puzzle)
        ...
        robot.contacts        # every time the robot touched the puzzle

    The meshes come from the STEP files (downloaded on first use, needs the
    ``step`` extra). The base plate is white with raised black marker cells,
    as printed, so the markers can be detected in ``robot.grab_image()``
    (``cv2.aruco``, DICT_4X4_50, IDs 1 and 2, 40 mm).

    The puzzle can rotate with a constant speed about the maze z axis (up)
    shifted to ``rotation_center``. The camera, the viewer and the contact
    checking all see it at the right time; the ground truth at time t is
    ``puzzle.pose_at(t)``.

    :param name: "A" - "E"
    :param pose: 4x4 pose of the maze frame (wire origin on the base plate
        top, z up) in the robot base frame, at the start of the rotation
    :param rotation_period: duration of one revolution [s], positive turns
        counter-clockwise seen from above, negative clockwise; None or 0
        for a puzzle standing still
    :param rotation_center: [x, y] point the rotation axis passes through,
        in the maze frame [m] (the wire origin by default)
    :param color: wire color
    :param base_color: base plate color
    :param marker_color: color of the raised marker cells
    :param spacing: camera surface sampling distance [m]
    """
    def __init__(self, name: str, pose: np.ndarray, color: int = 0xffa500,
                 base_color: int = 0xf0f0f0, marker_color: int = 0x202020,
                 spacing: float = 0.0015,
                 cache_dir: str | Path | None = None,
                 rotation_period: float | None = None,
                 rotation_center: ArrayLike = (0.0, 0.0)) -> None:
        from .visualizer import load_mesh

        if name not in "ABCDE" or len(name) != 1:
            raise ValueError(f"Unknown puzzle {name!r}, use A - E")
        self.name = name
        self.color, self.base_color, self.marker_color = (color, base_color,
                                                          marker_color)
        self.spacing = spacing
        self._pose = np.array(pose, dtype=float)
        self.rotation_period = rotation_period or None
        self.rotation_center = np.array([*rotation_center, 0.0][:3], dtype=float)
        #: Start of the rotation (time.monotonic() clock)
        self.t0 = time.monotonic()
        files = download_puzzles(cache_dir, names=[name, "base"])
        self.wire_mesh = load_mesh(files[name])
        base = load_mesh(files["base"])
        # the raised marker cells are the only faces above the plate top
        top = np.asarray(base.vertices)[np.asarray(base.faces)][:, :, 2].max(axis=1)
        raised = top > BASE_PLATE_THICKNESS + 1e-4
        self.base_mesh = base
        self._base_faces = (np.asarray(base.faces)[~raised],
                            np.asarray(base.faces)[raised])
        self._surfaces = None
        self._collision = None
        self._opening = None
        self._viz = None
        self._viz_name = None

    # -- placement ------------------------------------------------------
    @property
    def rotating(self) -> bool:
        return self.rotation_period is not None

    def pose_at(self, t: float | None = None) -> np.ndarray:
        """
        4x4 pose of the maze frame in the robot base frame at time t.
        :param t: time.monotonic() time, now if None
        """
        if not self.rotating:
            return self._pose.copy()
        t = time.monotonic() if t is None else t
        angle = 2 * np.pi * (t - self.t0) / self.rotation_period
        c = self.rotation_center
        return self._pose @ translation(c) @ rz(angle) @ translation(-c)

    @property
    def pose(self) -> np.ndarray:
        """4x4 pose of the maze frame in the robot base frame now. Setting
        it places the puzzle there and restarts the rotation from it."""
        return self.pose_at()

    @pose.setter
    def pose(self, pose: np.ndarray) -> None:
        self._pose = np.array(pose, dtype=float)
        self.t0 = time.monotonic()
        self._update_viz()

    def restart(self) -> None:
        """Start the rotation again from the initial pose."""
        self.t0 = time.monotonic()
        self._update_viz()

    def base_pose_at(self, t: float | None = None) -> np.ndarray:
        """4x4 pose of the base plate frame (bottom face center) at time t."""
        return self.pose_at(t) @ translation([0, 0, -BASE_PLATE_THICKNESS])

    @property
    def base_pose(self) -> np.ndarray:
        """4x4 pose of the base plate frame (bottom face center) now."""
        return self.base_pose_at()

    def _part_pose(self, part: str, t: float | None = None) -> np.ndarray:
        return self.pose_at(t) if part == "wire" else self.base_pose_at(t)

    def marker_poses(self, t: float | None = None) -> dict[int, np.ndarray]:
        """
        Ground truth marker poses: marker ID -> 4x4 pose of the marker center
        on its top face, axes as the maze frame (x right, y up in the marker
        image, z out of the plate).
        :param t: time.monotonic() time, now if None
        """
        base = self.base_pose_at(t)
        return {i: base @ translation([*xy, BASE_PLATE_THICKNESS
                                       + MARKER_HEIGHT])
                for i, xy in MARKER_CENTERS.items()}

    def marker_corners(self, t: float | None = None) -> dict[int, np.ndarray]:
        """
        Ground truth marker corners: marker ID -> 4x3 points in the robot base
        frame, in the ``cv2.aruco`` order (top-left, top-right, bottom-right,
        bottom-left of the marker image).
        :param t: time.monotonic() time, now if None
        """
        h = MARKER_SIZE / 2
        local = np.array([[-h, h, 0, 1], [h, h, 0, 1], [h, -h, 0, 1],
                          [-h, -h, 0, 1]])
        return {i: (p @ local.T).T[:, :3]
                for i, p in self.marker_poses(t).items()}

    # -- camera ---------------------------------------------------------
    def load(self) -> None:
        """Sample the surfaces for the camera (done on first use)."""
        if self._surfaces is not None:
            return
        from .camera import color_rgb, sample_mesh

        plate, cells = self._base_faces
        verts = np.asarray(self.base_mesh.vertices)
        self._surfaces = [
            ("wire", sample_mesh(self.wire_mesh.vertices, self.wire_mesh.faces,
                                 self.spacing, color_rgb(self.color))),
            ("base", sample_mesh(verts, plate, self.spacing,
                                 color_rgb(self.base_color))),
            ("markers", sample_mesh(verts, cells, self.spacing / 2,
                                    color_rgb(self.marker_color))),
        ]

    def surfaces_at(self, t: float):
        self.load()
        return [(s, self._part_pose(part, t)) for part, s in self._surfaces]

    # -- contacts -------------------------------------------------------
    def collision_geometries_at(self, t: float):
        """[(name, coal geometry, 4x4 pose), ...] for the contact checker."""
        if self._collision is None:
            from .contact import mesh_geometry

            self._collision = [
                ("wire", mesh_geometry(self.wire_mesh.vertices,
                                       self.wire_mesh.faces)),
                ("base", mesh_geometry(self.base_mesh.vertices,
                                       self.base_mesh.faces)),
            ]
        return [(f"puzzle_{self.name}/{part}", geom, self._part_pose(part, t))
                for part, geom in self._collision]

    def wire_through_hoop(self, hoop_pose: np.ndarray,
                          t: float | None = None) -> bool:
        """
        Whether the wire passes through the hoop opening, i.e. the hoop is
        threaded on the wire (touching it or not, see ``robot.contacts``).
        :param hoop_pose: 4x4 hoop center pose, ``robot.fk(q) @ HOOP_IN_FLANGE``
        :param t: time.monotonic() time the puzzle is posed at, now if None
        """
        import coal

        if self._opening is None:
            self._opening = coal.Cylinder(HOOP_INNER_RADIUS, 0.001)
        wire = self.collision_geometries_at(0.0)[0][1]
        tf = [coal.Transform3s(p[:3, :3], p[:3, 3])
              for p in (np.asarray(hoop_pose, dtype=float), self.pose_at(t))]
        return coal.collide(self._opening, tf[0], wire, tf[1],
                            coal.CollisionRequest(), coal.CollisionResult()) > 0

    # -- viewer ---------------------------------------------------------
    def attach(self, viz, name: str | None = None, fps: float = 30.0) -> None:
        """
        Draw the puzzle in the viewer (crs_sim.Visualizer), a rotating one
        keeps turning there.
        :param fps: update rate of the rotation in the viewer
        """
        import trimesh

        first = self._viz is None
        self._viz, self._viz_name = viz, name or f"puzzle_{self.name}"
        plate, cells = self._base_faces
        verts = np.asarray(self.base_mesh.vertices)
        viz.show_mesh(f"{self._viz_name}/wire", self.wire_mesh, self.pose,
                      color=self.color)
        viz.show_mesh(f"{self._viz_name}/base",
                      trimesh.Trimesh(verts, plate, process=False),
                      self.base_pose, color=self.base_color)
        viz.show_mesh(f"{self._viz_name}/markers",
                      trimesh.Trimesh(verts, cells, process=False),
                      self.base_pose, color=self.marker_color)
        if first and self.rotating:
            threading.Thread(target=self._animate, args=(1.0 / fps,),
                             daemon=True).start()

    def _update_viz(self) -> None:
        viz = self._viz
        if viz is not None:
            t = time.monotonic()
            for part in ("wire", "base", "markers"):
                viz.set_pose(f"{self._viz_name}/{part}", self._part_pose(part, t))

    def _animate(self, period: float) -> None:
        while self._viz is not None:
            self._update_viz()
            time.sleep(period)

    def detach(self) -> None:
        """Remove the puzzle from the viewer."""
        if self._viz is not None:
            self._viz.delete(self._viz_name)
            self._viz = None


def random_pose(rng: np.random.Generator | int | None = None,
                x: tuple[float, float] = (0.35, 0.55),
                y: tuple[float, float] = (-0.2, 0.2),
                yaw: tuple[float, float] = (-np.pi, np.pi),
                table_z: float = 0.0) -> np.ndarray:
    """
    Random puzzle placement on the table, uniform in the given ranges.
    :param rng: numpy Generator or seed
    :param table_z: height of the table top in the robot base frame [m]
    :return: 4x4 maze pose for :class:`Puzzle`
    """
    rng = np.random.default_rng(rng)
    return pose([rng.uniform(*x), rng.uniform(*y),
                 table_z + BASE_PLATE_THICKNESS], rng.uniform(*yaw))


def show_puzzle(viz, name: str, maze_pose: np.ndarray, cad: bool = True,
                path: bool = True, color: int = 0xffa500) -> None:
    """
    Draw a puzzle into the viewer only (no camera, no contacts, see
    :class:`Puzzle` and ``robot.add_object`` for that): base plate and wire
    from the STEP models (downloaded on first use) and/or the wire center line.
    :param viz: crs_sim.Visualizer (``robot.viz``)
    :param name: "A" - "E"
    :param maze_pose: 4x4 pose of the maze frame in the robot base frame
    :param cad: draw the CAD models (needs the ``step`` extra)
    :param path: draw the wire center line as points (if it is in MAZES)
    :param color: color of the wire
    """
    if cad:
        Puzzle(name, maze_pose, color=color).attach(viz)
    if path and name in MAZES:
        points = (maze_pose @ densify(maze_path(name), 0.01))[:, :3, 3]
        viz.show_points(f"puzzle_{name}/path", points, color=0xff0000,
                        radius=0.003)
