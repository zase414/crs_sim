"""
OpenGL renderer of the simulated camera (moderngl, headless through EGL).

Surfaces with a triangle mesh (``Surface.triangles``) are drawn as meshes,
surfaces given only as points are drawn as squares of about their sampling
spacing (like the numpy point renderer). Shading is the same headlight
Lambert as the point renderer's, the projection is the camera's K exactly
(pixel centers at integer coordinates, OpenCV convention).
"""
from __future__ import annotations

import threading
import weakref

import numpy as np

_VS = """
#version 330
uniform mat4 model;
uniform mat4 view_proj;
uniform float point_size;  // 1.5 * spacing * focal length, points only
in vec3 in_pos;
in vec3 in_normal;
in vec3 in_color;
out vec3 v_pos;
out vec3 v_normal;
out vec3 v_color;
void main() {
    vec4 p = model * vec4(in_pos, 1.0);
    v_pos = p.xyz;
    v_normal = mat3(model) * in_normal;
    v_color = in_color;
    gl_Position = view_proj * p;
    gl_PointSize = clamp(ceil(point_size / gl_Position.w), 1.0, 8.0);
}
"""

_FS = """
#version 330
uniform vec3 cam_pos;
in vec3 v_pos;
in vec3 v_normal;
in vec3 v_color;
out vec4 f_color;
void main() {
    float lambert = abs(dot(normalize(v_normal), normalize(cam_pos - v_pos)));
    f_color = vec4(v_color * (0.35 + 0.65 * lambert), 1.0);
}
"""


def projection(K: np.ndarray, width: int, height: int, near: float,
               far: float) -> np.ndarray:
    """
    OpenGL projection of the OpenCV camera frame through K. Image row 0
    (the top) is the first row read back from the framebuffer.
    """
    fx, fy, cx, cy, s = K[0, 0], K[1, 1], K[0, 2], K[1, 2], K[0, 1]
    w, h = width, height
    return np.array([
        [2 * fx / w, 2 * s / w, 2 * (cx + 0.5) / w - 1, 0],
        [0, 2 * fy / h, 2 * (cy + 0.5) / h - 1, 0],
        [0, 0, (far + near) / (far - near), -2 * far * near / (far - near)],
        [0, 0, 1, 0]])


class GLRenderer:
    """
    Renders surfaces seen by a camera with OpenGL. Raises on creation if no
    headless OpenGL (EGL) context can be created.
    """
    far = 20.0

    def __init__(self) -> None:
        import moderngl

        self._moderngl = moderngl
        # EGL only: GLX needs an X server and kills the process on errors
        self.ctx = moderngl.create_standalone_context(backend="egl")
        self.ctx.enable(moderngl.DEPTH_TEST | moderngl.PROGRAM_POINT_SIZE)
        self._prog = self.ctx.program(vertex_shader=_VS, fragment_shader=_FS)
        self._arrays = weakref.WeakKeyDictionary()  # Surface -> vertex array
        self._size = None
        self._lock = threading.Lock()

    def _framebuffers(self, width: int, height: int) -> None:
        if self._size == (width, height):
            return
        ctx = self.ctx
        try:   # multisampled, anti-aliased edges like a real camera's
            self._msaa = ctx.framebuffer(
                ctx.renderbuffer((width, height), samples=4),
                ctx.depth_renderbuffer((width, height), samples=4))
        except Exception:
            self._msaa = None
        self._fbo = ctx.framebuffer(ctx.renderbuffer((width, height)),
                                    ctx.depth_renderbuffer((width, height)))
        self._size = (width, height)

    def _vertex_array(self, surface):
        vao = self._arrays.get(surface)
        if vao is None:
            if surface.triangles is not None:
                tri = surface.triangles
                n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
                n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
                pos, nrm = tri.reshape(-1, 3), np.repeat(n, 3, axis=0)
                col = np.tile(surface.color, (len(pos), 1))
            else:
                pos, nrm, col = surface.points, surface.normals, surface.colors
            vbo = self.ctx.buffer(np.hstack([pos, nrm, col]).astype("f4").tobytes())
            vao = self.ctx.vertex_array(
                self._prog, [(vbo, "3f 3f 3f", "in_pos", "in_normal", "in_color")])
            self._arrays[surface] = vao
        return vao

    def render(self, camera, surfaces) -> np.ndarray:
        """
        :param camera: crs_sim.camera.Camera
        :param surfaces: list of (Surface, 4x4 pose in the base frame)
        :return: HxWx3 float32 RGB image in 0..255 (no noise)
        """
        w, h = camera.width, camera.height
        view = np.eye(4)
        view[:3, :3] = camera.pose[:3, :3].T
        view[:3, 3] = -camera.pose[:3, :3].T @ camera.pose[:3, 3]
        view_proj = projection(camera.K, w, h, camera.near, self.far) @ view
        prog = self._prog
        with self._lock, self.ctx:
            self._framebuffers(w, h)
            target = self._msaa or self._fbo
            target.use()
            self.ctx.clear(*(np.asarray(camera.background) / 255), 1.0)
            prog["view_proj"].write(view_proj.T.astype("f4").tobytes())
            prog["cam_pos"].value = tuple(camera.pose[:3, 3])
            for surface, pose in surfaces:
                vao = self._vertex_array(surface)
                prog["model"].write(np.asarray(pose, "f4").T.tobytes())
                if surface.triangles is not None:
                    vao.render(self._moderngl.TRIANGLES)
                else:
                    prog["point_size"].value = 1.5 * surface.spacing * camera.K[0, 0]
                    vao.render(self._moderngl.POINTS)
            if target is not self._fbo:
                self.ctx.copy_framebuffer(self._fbo, target)
            data = self._fbo.read(components=3)
        return np.frombuffer(data, np.uint8).reshape(h, w, 3).astype(np.float32)


_shared: GLRenderer | Exception | None = None


def shared_renderer() -> GLRenderer:
    """The renderer shared by all cameras, raises if OpenGL is not available."""
    global _shared
    if _shared is None:
        try:
            _shared = GLRenderer()
        except Exception as e:
            _shared = e
    if isinstance(_shared, Exception):
        raise _shared
    return _shared
