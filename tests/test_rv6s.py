import numpy as np
import pytest

from crs_sim import URDF_PATH_RV6S, Rv6s, Rv6sGripper
from crs_sim.puzzles import HOOP_IN_FLANGE


@pytest.fixture
def robot():
    r = Rv6s(visualize=False, camera=False, speedup=np.inf)
    r.initialize()
    return r


def test_fk_as_ctu_mitsubishi(robot):
    # ctu_mitsubishi tests/test_kinematics.py::test_fk0
    pose = robot.fk(np.deg2rad([0, 0, 90, 0, 0, 0]))
    np.testing.assert_allclose(pose[:3, 3], [0.485, 0, 0.73], atol=1e-6)
    np.testing.assert_allclose(pose[:3, :3], [[0, 0, 1], [0, 1, 0], [-1, 0, 0]],
                               atol=1e-6)


def test_ik_reaches_fk_pose(robot):
    rng = np.random.default_rng(0)
    for _ in range(200):
        q = rng.uniform(robot.q_min, robot.q_max)
        if abs(np.sin(q[4])) < 0.05:   # ZYZ gimbal lock of the real IK
            continue
        pose = robot.fk(q)
        sols = robot.ik(pose)
        assert any(np.allclose(s, q, atol=1e-6) for s in sols)
        for s in sols:
            assert np.allclose(robot.fk(s), pose, atol=1e-9)


def test_urdf_flange_is_fk(robot):
    pin = pytest.importorskip("pinocchio")
    model, _, _ = pin.buildModelsFromUrdf(str(URDF_PATH_RV6S),
                                          package_dirs=[str(URDF_PATH_RV6S.parent)])
    data = model.createData()
    rng = np.random.default_rng(1)
    for _ in range(50):
        q = rng.uniform(robot.q_min, robot.q_max)
        pin.framesForwardKinematics(model, data, q)
        assert np.allclose(data.oMf[model.getFrameId("flange")].homogeneous,
                           robot.fk(q), atol=1e-9)


def test_same_hoop_as_crs97(robot):
    assert np.allclose(robot.hoop_in_flange, HOOP_IN_FLANGE)


def test_needs_initialize_like_the_real_one():
    r = Rv6s(visualize=False, camera=False)
    with pytest.raises(AssertionError):
        r.get_q()
    with pytest.raises(AssertionError):
        r.move_to_q(r.q_home)
    r.initialize()
    assert np.allclose(r.get_q(), r.q_home)


def test_only_the_real_interface(robot):
    for name in ("in_motion", "set_speed", "release", "hard_home", "close"):
        assert not hasattr(robot, name), name


def test_move_is_non_blocking():
    r = Rv6s(visualize=False, camera=False, speedup=5.0)
    r.initialize()
    q = r.q_home + [0.3, 0.2, -0.2, 0.0, 0.3, 0.5]
    r.move_to_q(q)
    assert not np.allclose(r.get_q(), q, atol=1e-3)
    assert r.wait_for_motion_stop(timeout=0.01) is False
    assert r.wait_for_motion_stop() is True
    assert np.allclose(r.get_q(), q)


def test_move_rejects_joint_limits(robot):
    with pytest.raises(AssertionError):
        robot.move_to_q(robot.q_max + 0.1)


def test_soft_home_and_stop(robot):
    robot.move_to_q(robot.q_home + 0.2)
    robot.wait_for_motion_stop()
    robot.soft_home()
    robot.wait_for_motion_stop()
    assert np.allclose(robot.get_q(), robot.q_home)
    robot.stop_robot()
    robot.close_connection()


def test_gripper():
    g = Rv6sGripper()
    g.open()
    g.close()
    assert g.state == "closed"
    g.disconnect()
