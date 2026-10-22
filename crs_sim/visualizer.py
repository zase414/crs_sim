"""
MeshCat visualization of the robot (URDF + meshes) and of helper objects.
"""
from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike


class Visualizer:
    """
    Renders the URDF model in a MeshCat browser window and lets you draw
    helper objects (points, frames, boxes) into the scene.

    All helper objects are stored under ``objects/<name>`` and can be removed
    with :meth:`delete`.
    """
    def __init__(self, urdf_path: str | Path, open_browser: bool = True,
                 zmq_url: str | None = None) -> None:
        import pinocchio as pin
        from pinocchio.visualize import MeshcatVisualizer

        urdf_path = Path(urdf_path)
        model, collision_model, visual_model = pin.buildModelsFromUrdf(
            str(urdf_path), package_dirs=[str(urdf_path.parent)]
        )
        self._viz = MeshcatVisualizer(model, collision_model, visual_model)
        if zmq_url is None:
            self._viz.initViewer(open=open_browser)
        else:
            self._viz.initViewer(open=open_browser, zmq_url=zmq_url)
        self._viz.loadViewerModel(rootNodeName="crs")
        # meshcat's zmq socket is not thread-safe, the motion thread renders too
        self._lock = threading.Lock()
        self.set_camera([1.2, -1.4, 1.0])
        print(f"Simulator running at {self.url}")

    @property
    def url(self) -> str:
        """URL of the MeshCat web page."""
        return self._viz.viewer.url()

    @property
    def viewer(self):
        """Raw ``meshcat.Visualizer`` for anything not covered here."""
        return self._viz.viewer

    def set_camera(self, position: ArrayLike) -> None:
        """
        Place the camera, it keeps looking at the robot base.
        :param position: [x, y, z] in the robot base frame [m]
        """
        x, y, z = position
        with self._lock:
            # meshcat camera lives in a y-up frame
            self._viz.viewer["/Cameras/default/rotated/<object>"].set_property(
                "position", [float(x), float(z), float(-y)])

    def display(self, q: ArrayLike) -> None:
        """
        Show the robot in configuration q.
        :param q: joint configuration [rad]
        """
        with self._lock:
            self._viz.display(np.asarray(q, dtype=float))

    def show_points(self, name: str, points: ArrayLike, color: int = 0xffa500,
                    radius: float = 0.005) -> None:
        """
        Draw a set of spheres (e.g. a maze path).
        :param name: object name, redrawing under the same name replaces it
        :param points: Nx3 array of positions in the robot base frame [m]
        :param color: RGB hex color
        :param radius: sphere radius [m]
        """
        import meshcat.geometry as g
        import meshcat.transformations as tf

        with self._lock:
            node = self._viz.viewer["objects"][name]
            node.delete()
            material = g.MeshLambertMaterial(color=color)
            for i, p in enumerate(np.atleast_2d(points)):
                node[str(i)].set_object(g.Sphere(radius), material)
                node[str(i)].set_transform(tf.translation_matrix(p))

    def show_frame(self, name: str, pose: np.ndarray, scale: float = 0.1) -> None:
        """
        Draw a coordinate frame (x red, y green, z blue).
        :param name: object name, redrawing under the same name moves it
        :param pose: 4x4 homogeneous matrix
        :param scale: axis length [m]
        """
        import meshcat.geometry as g

        with self._lock:
            node = self._viz.viewer["objects"][name]
            node.set_object(g.triad(scale))
            node.set_transform(np.asarray(pose, dtype=float))

    def show_box(self, name: str, size: ArrayLike, pose: np.ndarray | None = None,
                 color: int = 0x808080, opacity: float = 0.5) -> None:
        """
        Draw a box, e.g. an obstacle or the table.
        :param name: object name
        :param size: [x, y, z] full lengths [m]
        :param pose: 4x4 homogeneous matrix of the box center
        :param color: RGB hex color
        :param opacity: 0 (invisible) .. 1 (opaque)
        """
        import meshcat.geometry as g

        with self._lock:
            node = self._viz.viewer["objects"][name]
            node.set_object(g.Box(list(size)), g.MeshLambertMaterial(
                color=color, opacity=opacity, transparent=opacity < 1))
            node.set_transform(np.eye(4) if pose is None
                               else np.asarray(pose, dtype=float))

    def show_cylinder(self, name: str, radius: float, length: float,
                      pose: np.ndarray | None = None, color: int = 0xe03c1e,
                      opacity: float = 1.0) -> None:
        """
        Draw a cylinder, e.g. a rod.
        :param name: object name
        :param radius: [m]
        :param length: [m]
        :param pose: 4x4 pose of the bottom face center, the axis is its z axis
        :param color: RGB hex color
        :param opacity: 0 (invisible) .. 1 (opaque)
        """
        import meshcat.geometry as g
        import meshcat.transformations as tf

        with self._lock:
            node = self._viz.viewer["objects"][name]
            # meshcat cylinders are centered and along y
            node["geometry"].set_object(g.Cylinder(length, radius),
                                        g.MeshLambertMaterial(
                color=color, opacity=opacity, transparent=opacity < 1))
            node["geometry"].set_transform(
                tf.translation_matrix([0, 0, length / 2])
                @ tf.rotation_matrix(np.pi / 2, [1, 0, 0]))
            node.set_transform(np.eye(4) if pose is None
                               else np.asarray(pose, dtype=float))

    def show_lines(self, name: str, segments: ArrayLike,
                   color: int = 0x000000) -> None:
        """
        Draw line segments.
        :param name: object name
        :param segments: Nx2x3 segment end points in the robot base frame [m]
        :param color: RGB hex color
        """
        import meshcat.geometry as g

        points = np.asarray(segments, dtype=float).reshape(-1, 3)
        with self._lock:
            self._viz.viewer["objects"][name].set_object(g.LineSegments(
                g.PointsGeometry(points.T.astype(np.float32)),
                g.LineBasicMaterial(color=color)))

    def set_pose(self, name: str, pose: np.ndarray) -> None:
        """
        Move an already drawn helper object.
        :param name: object name
        :param pose: 4x4 homogeneous matrix
        """
        with self._lock:
            self._viz.viewer["objects"][name].set_transform(
                np.asarray(pose, dtype=float))

    def show_mesh(self, name: str, mesh, pose: np.ndarray | None = None,
                  color: int = 0xb0b0b0, opacity: float = 1.0,
                  scale: float = 1.0) -> None:
        """
        Draw a mesh from a CAD file (STEP/STP, STL, OBJ, PLY, GLB, ...) or a
        ``trimesh.Trimesh``. Needs the ``step`` extra (trimesh + cascadio).
        :param name: object name
        :param mesh: file path or trimesh object
        :param pose: 4x4 homogeneous matrix of the mesh origin
        :param color: RGB hex color
        :param opacity: 0 (invisible) .. 1 (opaque)
        :param scale: scale of the vertices, e.g. 0.001 for STL in mm (STEP
            files are converted to meters automatically)
        """
        import meshcat.geometry as g

        if not hasattr(mesh, "faces"):
            mesh = load_mesh(mesh)
        geometry = g.TriangularMeshGeometry(
            np.asarray(mesh.vertices, dtype=float) * scale,
            np.asarray(mesh.faces, dtype=np.uint32))
        with self._lock:
            node = self._viz.viewer["objects"][name]
            node.set_object(geometry, g.MeshLambertMaterial(
                color=color, opacity=opacity, transparent=opacity < 1))
            node.set_transform(np.eye(4) if pose is None
                               else np.asarray(pose, dtype=float))

    def delete(self, name: str | None = None) -> None:
        """
        Remove a helper object.
        :param name: object to remove, all helper objects if None
        """
        with self._lock:
            if name is None:
                self._viz.viewer["objects"].delete()
            else:
                self._viz.viewer["objects"][name].delete()


def load_mesh(path: str | Path):
    """
    Load a CAD/mesh file as a single ``trimesh.Trimesh``. STEP files are
    tessellated by OpenCASCADE (cascadio) and returned in meters.
    """
    try:
        import trimesh
    except ImportError as e:
        raise ImportError(
            "Loading meshes needs the 'step' extra: pip install 'crs-sim[step]'"
        ) from e
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return trimesh.load(str(path), force="mesh")
