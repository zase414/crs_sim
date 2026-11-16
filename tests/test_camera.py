import time

import numpy as np
import pytest

from crs_sim import CRS97
from crs_sim.camera import Camera, look_at, write_png
from crs_sim.puzzles import HOOP_IN_FLANGE

Q_OVER_TABLE = np.array([0.200, -0.404, -2.008, 0.000, -0.729, 0.724])


@pytest.fixture
def robot():
    r = CRS97(visualize=False, speedup=np.inf,
              camera=Camera(width=400, height=300, noise=0.0, fps=None, lag=0.0))
    r.initialize()
    r.move_to_q(Q_OVER_TABLE)
    return r


def blue_centroid(image, near=None, radius=30):
    """Centroid of the hoop (blue) pixels of a BGR image, only those within
    radius of the pixel near if given."""
    b, r = image[:, :, 0].astype(int), image[:, :, 2].astype(int)
    v, u = np.nonzero((b > 100) & (b > r + 50))
    if near is not None:
        keep = np.hypot(u - near[0], v - near[1]) < radius
        u, v = u[keep], v[keep]
    assert len(u) > 0
    return np.array([u.mean(), v.mean()])


def test_project_backproject_roundtrip():
    cam = Camera(look_at([0.3, 0.1, 1.5], [0.4, -0.1, 0.0]))
    rng = np.random.default_rng(0)
    points = rng.uniform([0, -0.5, 0], [0.8, 0.5, 0.4], size=(50, 3))
    uv, depth = cam.project(points)
    assert np.all(depth > 0)
    for p, px in zip(points, uv):
        assert np.allclose(cam.backproject(px, p[2])[0], p, atol=1e-9)


def test_default_camera_looks_down():
    cam = Camera()
    assert cam.pose[:3, 2] @ [0, 0, -1] > np.cos(np.deg2rad(10))
    assert np.allclose(cam.pose[:3, :3].T @ cam.pose[:3, :3], np.eye(3),
                       atol=1e-5)
    # the workspace in front of the robot is in the image
    uv, _ = cam.project([[0.45, 0.0, 0.0]])
    assert 0 <= uv[0, 0] < cam.width and 0 <= uv[0, 1] < cam.height
    # base +x points up in the image
    uv, _ = cam.project([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    assert uv[1, 1] < uv[0, 1]


def test_grab_image_shows_hoop(robot):
    image = robot.grab_image()
    assert image.shape == (300, 400, 3) and image.dtype == np.uint8
    hoop = (robot.fk(robot.get_q()) @ HOOP_IN_FLANGE)[:3, 3]
    uv, _ = robot.camera.project(hoop)
    # the centroid is pulled a bit towards the hoop's stem
    assert np.linalg.norm(blue_centroid(image, uv[0]) - uv[0]) < 5


def test_camera_image_overrides_rendering(robot):
    robot.camera_image = np.zeros((4, 4, 3), np.uint8)
    assert robot.grab_image().shape == (4, 4, 3)


def test_no_camera():
    r = CRS97(visualize=False, camera=False)
    with pytest.raises(RuntimeError):
        r.grab_image()


def test_q_at_returns_past_configurations():
    r = CRS97(visualize=False, speedup=20.0, camera=False)
    r.initialize(home=False)
    t0 = time.monotonic()
    r.move_to_q(r.q_home + [0.3, 0, 0, 0, 0, 0])
    r.wait_for_motion_stop()
    t1 = time.monotonic()
    r.move_to_q(r.q_home)
    r.wait_for_motion_stop()
    assert np.allclose(r.q_at(t0 - 1.0), r.q_home, atol=1e-4)
    assert np.allclose(r.q_at(t1), r.q_home + [0.3, 0, 0, 0, 0, 0], atol=1e-4)
    assert np.allclose(r.q_at(time.monotonic()), r.get_q())


def test_lagged_image_shows_past_pose():
    r = CRS97(visualize=False, speedup=np.inf,
              camera=Camera(width=400, height=300, noise=0.0, fps=None, lag=0.3))
    r.initialize()
    r.move_to_q(Q_OVER_TABLE)
    time.sleep(0.5)
    before = r.grab_image()
    r.move_to_q(Q_OVER_TABLE + [0.15, 0, 0, 0, 0, 0])
    lagged = r.grab_image()   # still the old pose
    assert r.camera.last_exposure_time < time.monotonic() - 0.3
    assert np.linalg.norm(blue_centroid(lagged) - blue_centroid(before)) < 1
    time.sleep(0.4)
    after = r.grab_image()
    assert np.linalg.norm(blue_centroid(after) - blue_centroid(before)) > 10


def test_lag_is_delivered_exactly():
    r = CRS97(visualize=False, speedup=np.inf,
              camera=Camera(width=200, height=150, fps=None, lag=0.2))
    for lag in (0.2, 0.4):
        r.camera.lag = lag
        r.grab_image()  # calibrates the rendering time estimate
        r.grab_image()
        age = time.monotonic() - r.camera.last_exposure_time
        assert lag - 1e-3 < age < lag + 0.05


def test_fps_quantizes_exposure():
    r = CRS97(visualize=False, camera=Camera(width=80, height=60, fps=10.0))
    r.grab_image()
    t = r.camera.last_exposure_time
    assert abs(t * 10 - round(t * 10)) < 1e-6


def test_on_grab_gets_every_image(robot):
    grabbed = []
    robot.camera.on_grab.append(lambda image, t: grabbed.append((image, t)))
    image = robot.grab_image()
    assert len(grabbed) == 1 and grabbed[0][0] is image
    assert grabbed[0][1] == robot.camera.last_exposure_time


@pytest.mark.parametrize("renderer", ["gl", "points"])
def test_renderers_agree(renderer):
    if renderer == "gl":
        pytest.importorskip("moderngl")
        from crs_sim.gl_render import shared_renderer
        try:
            shared_renderer()
        except Exception as e:
            pytest.skip(f"no OpenGL: {e}")
    r = CRS97(visualize=False, speedup=np.inf, camera=Camera(
        width=400, height=300, noise=0.0, fps=None, lag=0.0, renderer=renderer))
    r.initialize()
    r.move_to_q(Q_OVER_TABLE)
    image = r.camera.render()
    hoop = (r.fk(r.get_q()) @ HOOP_IN_FLANGE)[:3, 3]
    uv, _ = r.camera.project(hoop)
    assert np.linalg.norm(blue_centroid(image, uv[0]) - uv[0]) < 5


def test_write_png(tmp_path):
    image = np.zeros((5, 7, 3), np.uint8)
    image[1, 2] = [255, 0, 0]
    path = tmp_path / "a.png"
    write_png(path, image)
    data = path.read_bytes()
    assert data.startswith(b"\x89PNG") and data.endswith(b"IEND\xaeB`\x82")
