"""
Simulated overhead camera: renders the robot, the hoop and any added objects
(e.g. a :class:`crs_sim.rod.MovingRod`) into an image, with a configurable
input lag.

The image at grab time ``t`` shows the scene as it was at ``t - lag``
(optionally plus random jitter and quantized to the camera frame rate), like a
real camera whose frames reach your code late. The robot and the objects keep
a short history of their motion, so a lagged frame is rendered exactly as the
scene was at its exposure time.

Rendering is a small numpy point-splatting renderer: object surfaces are
sampled into points (with normals and colors) once, then every frame the
points are transformed, projected through a pinhole model, shaded (headlight
Lambert) and z-buffered. It is meant for testing perception pipelines (color
segmentation, circle/ellipse fitting, back-projection), not photorealism.

Frames: the camera frame follows the OpenCV convention (x right, y down,
z along the optical axis), the camera pose is given in the robot base frame.
Times are ``time.monotonic()`` seconds.
"""
from __future__ import annotations

import math
import struct
import time
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike


@dataclass
class Surface:
    """Points sampled on an object surface, in the object frame."""
    points: np.ndarray   # Nx3 [m]
    normals: np.ndarray  # Nx3 unit
    colors: np.ndarray   # Nx3 RGB in 0..1
    spacing: float       # approximate distance between neighbouring points [m]


def look_at(eye: ArrayLike, target: ArrayLike,
            up: ArrayLike = (1.0, 0.0, 0.0)) -> np.ndarray:
    """
    Camera pose (OpenCV convention) at ``eye`` looking at ``target``.
    :param eye: camera position in the base frame [m]
    :param target: point the optical axis passes through [m]
    :param up: direction that appears "up" in the image
    :return: 4x4 pose of the camera in the base frame
    """
    eye = np.asarray(eye, dtype=float)
    z = np.asarray(target, dtype=float) - eye
    z /= np.linalg.norm(z)
    y = -np.asarray(up, dtype=float)
    y = y - y.dot(z) * z
    if np.linalg.norm(y) < 1e-9:
        raise ValueError("up is parallel to the viewing direction")
    y /= np.linalg.norm(y)
    pose = np.eye(4)
    pose[:3, :3] = np.column_stack([np.cross(y, z), y, z])
    pose[:3, 3] = eye
    return pose


#: Default camera pose: 1.6 m above the base, looking straight down at the
#: workspace in front of the robot. The robot base is at the bottom of the
#: image, base +x points up in the image.
DEFAULT_CAMERA_POSE = look_at([0.45, 0.0, 1.6], [0.45, 0.0, 0.0])


class Camera:
    """
    Pinhole camera looking at the simulated scene.

    :param pose: 4x4 camera pose in the robot base frame (OpenCV convention),
        :data:`DEFAULT_CAMERA_POSE` if None
    :param width: image width [px]
    :param height: image height [px]
    :param fov: horizontal field of view [rad]
    :param lag: input lag [s], the image shows the scene ``lag`` seconds ago.
        Can be changed at any time (``robot.camera.lag = 0.2``).
    :param lag_jitter: standard deviation of a random extra lag [s] (only
        positive extra lag is added)
    :param fps: frame rate, exposure times are quantized to ``1 / fps``;
        None for continuous exposure times
    :param noise: standard deviation of the pixel noise [intensity levels]
    :param background: RGB color of everything not covered by an object
    :param bgr: return images in BGR channel order (OpenCV convention)
    :param seed: seed of the noise / jitter generator
    """
    def __init__(self, pose: np.ndarray | None = None, width: int = 2448,
                 height: int = 2050, fov: float = np.deg2rad(60),
                 lag: float = 0.1, lag_jitter: float = 0.0,
                 fps: float | None = 30.0, noise: float = 2.0,
                 background: ArrayLike = (60, 60, 60), bgr: bool = True,
                 seed: int | None = None) -> None:
        self.pose = np.array(DEFAULT_CAMERA_POSE if pose is None else pose,
                             dtype=float)
        self.width = int(width)
        self.height = int(height)
        self.fov = float(fov)
        self.lag = float(lag)
        self.lag_jitter = float(lag_jitter)
        self.fps = fps
        self.noise = float(noise)
        self.background = np.asarray(background, dtype=float)
        self.bgr = bgr
        self.near = 0.02
        #: Rendered objects, anything with ``surfaces_at(t)`` returning a
        #: list of (Surface, 4x4 pose in the base frame)
        self.objects: list = []
        #: Exposure time of the last grabbed frame (time.monotonic() clock)
        self.last_exposure_time: float | None = None
        self._rng = np.random.default_rng(seed)
        self._render_estimate = 0.0

    # ------------------------------------------------------------------
    # Intrinsics / geometry
    # ------------------------------------------------------------------
    @property
    def K(self) -> np.ndarray:
        """3x3 intrinsic matrix."""
        f = self.width / 2 / math.tan(self.fov / 2)
        return np.array([[f, 0, (self.width - 1) / 2],
                         [0, f, (self.height - 1) / 2],
                         [0, 0, 1]])

    def project(self, points: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        """
        Project points to the image.
        :param points: Nx3 points in the base frame [m]
        :return: (Nx2 pixel coordinates [u, v], N depths along the optical axis)
        """
        p = np.atleast_2d(np.asarray(points, dtype=float))
        pc = (p - self.pose[:3, 3]) @ self.pose[:3, :3]
        uvw = pc @ self.K.T
        return uvw[:, :2] / uvw[:, 2:3], pc[:, 2]

    def rays(self, uv: ArrayLike) -> np.ndarray:
        """
        Viewing rays through pixels.
        :param uv: Nx2 pixel coordinates
        :return: Nx3 unit directions in the base frame (from the camera center,
            which is ``pose[:3, 3]``)
        """
        uv = np.atleast_2d(np.asarray(uv, dtype=float))
        d = np.column_stack([uv, np.ones(len(uv))]) @ np.linalg.inv(self.K).T
        d = d @ self.pose[:3, :3].T
        return d / np.linalg.norm(d, axis=1, keepdims=True)

    def backproject(self, uv: ArrayLike, z: float) -> np.ndarray:
        """
        Intersect the viewing rays through pixels with the plane at height z.
        :param uv: Nx2 pixel coordinates
        :param z: plane height in the base frame [m]
        :return: Nx3 points in the base frame
        """
        d = self.rays(uv)
        c = self.pose[:3, 3]
        return c + d * ((z - c[2]) / d[:, 2])[:, None]

    # ------------------------------------------------------------------
    # Capturing
    # ------------------------------------------------------------------
    def add(self, obj) -> None:
        """Add an object (with ``surfaces_at(t)``) to the rendered scene."""
        if obj not in self.objects:
            self.objects.append(obj)

    def remove(self, obj) -> None:
        """Remove an object from the rendered scene."""
        self.objects.remove(obj)

    def grab(self) -> np.ndarray:
        """
        Lagged camera image. Like a real camera, it returns the newest frame
        delivered so far: a frame exposed at ``t_exp`` (a multiple of
        ``1 / fps``) is delivered at ``t_exp + lag`` (+ jitter).

        Rendering takes time (~0.1 s), so the call blocks for about that long,
        like waiting for a frame. The exposure time is chosen so that the
        frame is delivered when the call returns, i.e. the image is exactly
        ``lag`` old (+ up to one frame period). A lag shorter than the
        rendering time cannot be achieved, the image is then as old as the
        rendering took.

        :return: HxWx3 uint8 image, its exposure time is in
            ``last_exposure_time``
        """
        for obj in self.objects:
            if hasattr(obj, "load"):
                obj.load()  # first-use loading must not count as rendering time
        t_call = time.monotonic()
        lag = self.lag
        if self.lag_jitter > 0:
            lag += abs(self._rng.normal(0.0, self.lag_jitter))
        t_exp = min(t_call + self._render_estimate - lag, t_call)
        if self.fps:
            t_exp = math.floor(t_exp * self.fps) / self.fps
        image = self.render(t_exp)
        done = time.monotonic()
        # slow estimate decrease, fast increase
        self._render_estimate = max(done - t_call,
                                    0.9 * self._render_estimate)
        if done < t_exp + lag:
            time.sleep(t_exp + lag - done)
        self.last_exposure_time = t_exp
        return image

    def render(self, t: float | None = None) -> np.ndarray:
        """
        Render the scene as it was at time t (no lag applied).
        :param t: scene time (time.monotonic() clock), now if None
        :return: HxWx3 uint8 image (BGR if ``self.bgr``)
        """
        t = time.monotonic() if t is None else t
        h, w = self.height, self.width
        img = np.empty((h * w, 3), dtype=np.float32)
        img[:] = self.background

        pts, nrm, col, spc = [], [], [], []
        for obj in self.objects:
            for surface, pose in obj.surfaces_at(t):
                pts.append(surface.points @ pose[:3, :3].T + pose[:3, 3])
                nrm.append(surface.normals @ pose[:3, :3].T)
                col.append(surface.colors)
                spc.append(np.full(len(surface.points), surface.spacing))
        if pts:
            self._splat(img, np.concatenate(pts), np.concatenate(nrm),
                        np.concatenate(col), np.concatenate(spc))

        if self.noise > 0:
            img += self.noise * self._rng.standard_normal(img.shape,
                                                          dtype=np.float32)
        img = np.clip(img, 0, 255).astype(np.uint8).reshape(h, w, 3)
        return img[:, :, ::-1].copy() if self.bgr else img

    def _splat(self, img: np.ndarray, points: np.ndarray, normals: np.ndarray,
               colors: np.ndarray, spacing: np.ndarray) -> None:
        """Project, shade and z-buffer surface points into the flat image."""
        h, w = self.height, self.width
        # back-face culling (outward normals), then project
        to_cam = self.pose[:3, 3] - points
        facing = np.einsum("ij,ij->i", normals, to_cam)
        keep = facing > 0
        uv, depth = self.project(points[keep])
        lambert = facing[keep] / np.linalg.norm(to_cam[keep], axis=1)
        keep[keep] = depth > self.near
        front = depth > self.near
        uv, depth, lambert = uv[front], depth[front], lambert[front]
        shaded = colors[keep] * 255 * (0.35 + 0.65 * lambert)[:, None]

        # each point covers about (spacing projected) pixels, splat it as a
        # square of that size so that no holes appear between the points
        f = self.K[0, 0]
        size = np.clip(np.ceil(1.5 * spacing[keep] * f / depth), 1, 8).astype(int)
        us, vs, ds, idx = [], [], [], []
        base = np.floor(uv).astype(int)
        for s in np.unique(size):
            sel = np.flatnonzero(size == s)
            for du in range(-(s // 2), s - s // 2):
                for dv in range(-(s // 2), s - s // 2):
                    us.append(base[sel, 0] + du)
                    vs.append(base[sel, 1] + dv)
                    ds.append(depth[sel])
                    idx.append(sel)
        if not us:
            return
        u, v = np.concatenate(us), np.concatenate(vs)
        d, i = np.concatenate(ds), np.concatenate(idx)
        inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        flat = v[inside] * w + u[inside]
        d, i = d[inside], i[inside]
        # z-buffer: for every pixel keep the nearest point
        zbuf = np.full(h * w, np.inf)
        np.minimum.at(zbuf, flat, d)
        nearest = d <= zbuf[flat]
        img[flat[nearest]] = shaded[i[nearest]]

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------
    def frustum_segments(self, depth: float = 0.3) -> np.ndarray:
        """Line segments (Nx2x3, base frame) of the viewing frustum."""
        corners = np.array([[0, 0], [self.width - 1, 0],
                            [self.width - 1, self.height - 1],
                            [0, self.height - 1]], dtype=float)
        d = self.rays(corners)
        d *= depth / (d @ self.pose[:3, 2])[:, None]
        c = self.pose[:3, 3]
        far = c + d
        segments = [(c, p) for p in far]
        segments += [(far[k], far[(k + 1) % 4]) for k in range(4)]
        return np.array(segments)

    def show(self, viz, name: str = "camera", depth: float = 0.3) -> None:
        """
        Draw the camera frame and its viewing frustum.
        :param viz: crs_sim.Visualizer (``robot.viz``)
        :param depth: length of the drawn frustum [m]
        """
        viz.show_frame(f"{name}/frame", self.pose, scale=0.1)
        viz.show_lines(f"{name}/frustum", self.frustum_segments(depth),
                       color=0x202020)


# ----------------------------------------------------------------------
# Surface sampling
# ----------------------------------------------------------------------
def read_stl(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Read a binary or ASCII STL file.
    :return: (Vx3 vertices, Fx3 faces), vertices not merged
    """
    data = Path(path).read_bytes()
    n = struct.unpack_from("<I", data, 80)[0] if len(data) >= 84 else -1
    if len(data) == 84 + 50 * n:
        rec = np.frombuffer(data, dtype=np.dtype([
            ("normal", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")]),
            count=n, offset=84)
        vertices = rec["v"].reshape(-1, 3).astype(float)
    else:
        words = data.decode("ascii", errors="ignore").split()
        vertices = np.array([words[k + 1:k + 4] for k, wd in enumerate(words)
                             if wd == "vertex"], dtype=float)
    return vertices, np.arange(len(vertices)).reshape(-1, 3)


def sample_mesh(vertices: np.ndarray, faces: np.ndarray, spacing: float,
                color: ArrayLike) -> Surface:
    """
    Sample points on a triangle mesh on a regular grid per triangle, at most
    spacing apart (vertices and edges included, so there are no gaps between
    neighbouring triangles).
    :param color: RGB in 0..1
    """
    tri = np.asarray(vertices, dtype=float)[np.asarray(faces)]
    # put the vertex opposite the longest edge first, the grid then spans
    # the two shorter edges
    opposite = np.linalg.norm(tri[:, [1, 2, 0]] - tri[:, [2, 0, 1]], axis=2)
    order = (np.argmax(opposite, axis=1)[:, None] + np.arange(3)) % 3
    tri = tri[np.arange(len(tri))[:, None], order]
    o, e1, e2 = tri[:, 0], tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    normal = np.cross(e1, e2)
    valid = np.linalg.norm(normal, axis=1) > 1e-12
    o, e1, e2, normal = o[valid], e1[valid], e2[valid], normal[valid]
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)

    m = np.ceil(np.column_stack([np.linalg.norm(e1, axis=1),
                                 np.linalg.norm(e2, axis=1)]) / spacing)
    m = np.maximum(m, 1).astype(int)
    pairs, group = np.unique(m, axis=0, return_inverse=True)
    points, normals = [], []
    for g, (m1, m2) in enumerate(pairs):
        a, b = np.meshgrid(np.arange(m1 + 1) / m1, np.arange(m2 + 1) / m2)
        inside = a + b <= 1 + 1e-9
        a, b = a[inside], b[inside]
        sel = np.flatnonzero(group.ravel() == g)
        p = (o[sel, None] + a[None, :, None] * e1[sel, None]
             + b[None, :, None] * e2[sel, None])
        points.append(p.reshape(-1, 3))
        normals.append(np.repeat(normal[sel], len(a), axis=0))
    points, normals = np.concatenate(points), np.concatenate(normals)
    # shared vertices and edges were sampled once per triangle
    _, first = np.unique(np.round(points / (spacing / 20)), axis=0,
                         return_index=True)
    points, normals = points[first], normals[first]
    return Surface(points, normals,
                   np.tile(np.asarray(color, float), (len(points), 1)), spacing)


def cylinder_surface(radius: float, length: float, spacing: float,
                     color: ArrayLike) -> Surface:
    """
    Points on a closed cylinder, axis along z from z = 0 to z = length.
    :param color: RGB in 0..1
    """
    na = max(8, int(math.ceil(2 * math.pi * radius / spacing)))
    nz = max(2, int(math.ceil(length / spacing)) + 1)
    a = np.linspace(0, 2 * np.pi, na, endpoint=False)
    aa, zz = np.meshgrid(a, np.linspace(0, length, nz))
    side_n = np.column_stack([np.cos(aa.ravel()), np.sin(aa.ravel()),
                              np.zeros(aa.size)])
    side_p = side_n * radius
    side_p[:, 2] = zz.ravel()

    caps_p, caps_n = [], []
    for r in np.arange(0, radius, spacing):
        k = max(1, int(math.ceil(2 * math.pi * r / spacing)))
        b = np.linspace(0, 2 * np.pi, k, endpoint=False)
        ring = np.column_stack([r * np.cos(b), r * np.sin(b), np.zeros(k)])
        for z, nz_ in ((0.0, -1.0), (length, 1.0)):
            p = ring.copy()
            p[:, 2] = z
            caps_p.append(p)
            caps_n.append(np.tile([0.0, 0.0, nz_], (k, 1)))
    points = np.concatenate([side_p] + caps_p)
    normals = np.concatenate([side_n] + caps_n)
    return Surface(points, normals,
                   np.tile(np.asarray(color, float), (len(points), 1)), spacing)


def color_rgb(color: int) -> np.ndarray:
    """0xRRGGBB -> RGB in 0..1"""
    return np.array([(color >> 16) & 255, (color >> 8) & 255, color & 255]) / 255


# ----------------------------------------------------------------------
# The robot as seen by the camera
# ----------------------------------------------------------------------
class RobotBody:
    """
    The robot links (URDF visual meshes, the hoop included) for the camera,
    posed by the robot's joint history (``robot.q_at(t)``).
    :param robot: crs_sim.CRSRobot
    :param spacing: surface sampling distance of the links [m]
    :param hoop_spacing: finer sampling distance of the hoop [m]
    """
    def __init__(self, robot, spacing: float = 0.003,
                 hoop_spacing: float = 0.001) -> None:
        self.robot = robot
        self.spacing = spacing
        self.hoop_spacing = hoop_spacing
        self._surfaces = None  # loaded on first use, pinocchio is imported then

    def load(self) -> None:
        """Load the meshes (done on first use)."""
        if self._surfaces is not None:
            return
        import pinocchio as pin

        self._pin = pin
        urdf = Path(self.robot.urdf_path)
        self._model, _, self._visual = pin.buildModelsFromUrdf(
            str(urdf), package_dirs=[str(urdf.parent)])
        self._data = self._model.createData()
        self._visual_data = pin.GeometryData(self._visual)
        surfaces = []
        for k, geom in enumerate(self._visual.geometryObjects):
            vertices, faces = read_stl(geom.meshPath)
            vertices = vertices * np.asarray(geom.meshScale)
            fine = geom.name.startswith("hoop")
            surfaces.append(sample_mesh(
                vertices, faces, self.hoop_spacing if fine else self.spacing,
                np.asarray(geom.meshColor[:3])))
        self._surfaces = surfaces

    def surfaces_at(self, t: float) -> list[tuple[Surface, np.ndarray]]:
        self.load()
        pin = self._pin
        q = self.robot.q_at(t)
        pin.forwardKinematics(self._model, self._data, q)
        pin.updateGeometryPlacements(self._model, self._data, self._visual,
                                     self._visual_data)
        return [(s, self._visual_data.oMg[k].homogeneous)
                for k, s in enumerate(self._surfaces)]


def write_png(path: str | Path, image: np.ndarray, bgr: bool = True) -> None:
    """
    Save an HxWx3 uint8 image as PNG (no extra dependencies).
    :param bgr: the image is in BGR channel order
    """
    image = np.ascontiguousarray(image[:, :, ::-1] if bgr else image,
                                 dtype=np.uint8)
    h, w, _ = image.shape
    raw = b"".join(b"\x00" + image[r].tobytes() for r in range(h))

    def chunk(tag: bytes, payload: bytes) -> bytes:
        return (struct.pack(">I", len(payload)) + tag + payload
                + struct.pack(">I", zlib.crc32(tag + payload) & 0xffffffff))

    Path(path).write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 6))
        + chunk(b"IEND", b""))
