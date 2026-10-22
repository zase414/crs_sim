import time

import numpy as np
import pytest

from crs_sim import CRS97


@pytest.fixture
def robot():
    r = CRS97(visualize=False, speedup=np.inf)
    r.initialize()
    return r


def test_starts_at_home(robot):
    assert np.allclose(robot.get_q(), robot.q_home, atol=1e-4)


def test_ik_reaches_fk_pose(robot):
    rng = np.random.default_rng(0)
    for _ in range(200):
        q = rng.uniform(robot.q_min, robot.q_max)
        # Near the wrist singularity the (real robot's) IK snaps to q[4] = 0
        if abs(q[4]) < 0.05:
            continue
        pose = robot.fk(q)
        sols = robot.ik(pose)
        assert sols
        for s in sols:
            assert np.allclose(robot.fk(s), pose, atol=1e-9)


def test_move_requires_initialize():
    r = CRS97(visualize=False)
    with pytest.raises(AssertionError):
        r.move_to_q(r.q_home)


def test_move_rejects_joint_limits(robot):
    with pytest.raises(AssertionError):
        robot.move_to_q(robot.q_max + 0.1)


def test_move_is_non_blocking():
    r = CRS97(visualize=False, speedup=20.0)
    r.initialize(home=False)
    q = r.q_home + [0.3, 0.2, 0.0, 0.0, 0.0, 0.5]
    r.move_to_q(q)
    assert r.in_motion()
    assert not np.allclose(r.get_q(), q, atol=1e-3)
    r.wait_for_motion_stop()
    assert not r.in_motion()
    # quantized to encoder resolution
    assert np.allclose(r.get_q(), q, atol=1e-4)


def test_speed_changes_duration():
    r = CRS97(visualize=False, speedup=20.0)
    r.initialize(home=False)
    durations = []
    for fraction in (1.0, 0.0):
        r.set_speed_relative(fraction)
        t = time.monotonic()
        r.move_to_q(r.q_home + np.array([0.2, 0, 0, 0, 0, 0]) * (1 if fraction else -1))
        r.wait_for_motion_stop()
        durations.append(time.monotonic() - t)
    assert durations[1] > 2 * durations[0]

