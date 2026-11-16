import numpy as np
import pytest

from crs_sim.puzzles import MAZES, maze_path
from crs_sim.transforms import densify, interpolate, rx, rz, so3_exp, so3_log


def test_so3_log_exp_roundtrip():
    rng = np.random.default_rng(0)
    for _ in range(500):
        v = rng.normal(size=3)
        v *= rng.uniform(0, np.pi - 1e-3) / np.linalg.norm(v)
        assert np.allclose(so3_log(so3_exp(v)), v, atol=1e-9)
    for r in (rx(np.pi), rz(np.pi) @ rx(np.pi)):
        assert np.allclose(so3_exp(so3_log(r[:3, :3])), r[:3, :3], atol=1e-9)


def test_interpolate_endpoints():
    a, b = rz(0.3), rx(1.0)
    b[:3, 3] = [0.1, 0.2, 0.3]
    assert np.allclose(interpolate(a, b, 0), a)
    assert np.allclose(interpolate(a, b, 1), b)


@pytest.mark.parametrize("name", sorted(MAZES))
def test_maze_path(name):
    path = maze_path(name)
    # valid rigid transforms, starting at the base plate
    for p in path:
        assert np.allclose(p[:3, :3] @ p[:3, :3].T, np.eye(3), atol=1e-9)
    assert path[0, 2, 3] < 0.03
    assert path[:, 2, 3].max() > 0.2
    dense = densify(path, 0.005)
    steps = np.linalg.norm(np.diff(dense[:, :3, 3], axis=0), axis=1)
    assert steps.max() <= 0.005 + 1e-9
