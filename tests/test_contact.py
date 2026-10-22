import time
import urllib.error

import numpy as np
import pytest

from crs_sim import CRS97
from crs_sim.camera import Camera
from crs_sim.contact import ContactError
from crs_sim.puzzles import HOOP_IN_FLANGE, MARKER_SIZE
from crs_sim.rod import MovingRod
from crs_sim.transforms import pose, translation

# Puzzle A at MAZE_POSE, the hoop moved from 10 cm above the wire tip down to
# 2 cm above the base plate (examples/puzzle_a.py)
MAZE_POSE = pose([0.45, 0.0, 0.031])
DESCENT = np.array([
    [0.200, -0.280, -1.976, 0.000, -0.886, 0.724],
    [0.200, -0.404, -2.008, 0.000, -0.729, 0.724],
    [0.200, -0.548, -2.013, 0.000, -0.580, 0.724],
    [0.200, -0.706, -1.991, 0.000, -0.445, 0.724],
    [0.200, -0.873, -1.941, 0.000, -0.327, 0.724],
    [0.200, -1.037, -1.868, 0.000, -0.237, 0.724],
])
Q_GOAL = DESCENT[0]


def make_robot(**kwargs):
    r = CRS97(visualize=False, camera=False, speedup=np.inf, **kwargs)
    r.initialize()
    return r


def rod_through_hoop(robot, q):
    """Static rod thicker than the hoop opening, at the hoop center at q."""
    center = (robot.fk(q) @ HOOP_IN_FLANGE)[:3, 3]
    return MovingRod(translation(center - [0, 0, 0.05]), radius=0.05, length=0.1)


@pytest.fixture(scope="module")
def puzzle_a():
    pytest.importorskip("trimesh")
    from crs_sim.puzzles import Puzzle
    try:
        return Puzzle("A", MAZE_POSE)
    except urllib.error.URLError as e:
        pytest.skip(f"puzzle models not available: {e}")


def test_rod_contact_detected():
    robot = make_robot(on_contact="ignore")
    robot.add_object(rod_through_hoop(robot, Q_GOAL))
    assert robot.contacts == []
    robot.move_to_q(Q_GOAL)
    robot.wait_for_motion_stop()
    assert len(robot.contacts) == 1
    event = robot.contacts[0]
    assert "hoop" in [c.link for c in event.contacts]
    assert event.contacts[0].obstacle == "rod"
    assert robot.in_contact()
    robot.clear_contacts()
    assert robot.contacts == []


def test_contact_before_add_is_not_checked():
    robot = make_robot(on_contact="ignore")
    robot.move_to_q(Q_GOAL)
    robot.wait_for_motion_stop()
    robot.move_to_q(robot.q_home)
    robot.wait_for_motion_stop()
    robot.add_object(rod_through_hoop(robot, Q_GOAL))
    assert robot.contacts == []


def test_interrupted_motion_is_checked_only_up_to_where_it_got():
    robot = make_robot(on_contact="ignore")
    robot.speedup = 1.0
    robot.add_object(rod_through_hoop(robot, Q_GOAL))
    robot.move_to_q(Q_GOAL)
    duration = robot._motion.duration
    assert duration > 1.0
    time.sleep(0.2 * duration)
    robot.move_to_q(robot.q_home)  # turn back long before reaching the rod
    robot.wait_for_motion_stop()
    assert robot.contacts == []


def test_raise_on_contact():
    robot = make_robot(on_contact="raise")
    robot.add_object(rod_through_hoop(robot, Q_GOAL))
    robot.move_to_q(Q_GOAL)
    with pytest.raises(ContactError) as e:
        robot.wait_for_motion_stop()
    assert e.value.event is robot.contacts[0]
    robot.move_to_q(robot.q_home)  # raised once only
    robot.wait_for_motion_stop()


def test_puzzle_a_descent_is_free_and_a_side_step_touches(puzzle_a):
    robot = make_robot(on_contact="ignore")
    robot.add_object(puzzle_a)
    for q in list(DESCENT) + list(DESCENT[::-1]):
        robot.move_to_q(q)
        robot.wait_for_motion_stop()
    assert robot.contacts == []
    hoop_gap = robot.contact_checker.distance(DESCENT[3], links=["hoop"])
    assert 0.01 < hoop_gap < 0.03

    q = DESCENT[3].copy()
    q[0] += 0.08  # hoop pushed sideways through the wire
    robot.move_to_q(q)
    robot.wait_for_motion_stop()
    assert len(robot.contacts) == 1
    assert {(c.link, c.obstacle) for c in robot.contacts[0].contacts} == {
        ("hoop", "puzzle_A/wire")}


def test_puzzle_markers(puzzle_a):
    corners = puzzle_a.marker_corners()
    assert sorted(corners) == [1, 2]
    for c in corners.values():
        sides = np.linalg.norm(np.diff(c, axis=0, append=c[:1]), axis=1)
        assert np.allclose(sides, MARKER_SIZE)
        # cv2.aruco order seen from above: clockwise
        assert np.cross(c[1] - c[0], c[2] - c[1])[2] < 0

    # dark marker border and white plate in the camera image
    cam = Camera(width=800, height=600, noise=0.0, fps=None, lag=0.0)
    cam.add(puzzle_a)
    image = cam.render()
    gray = image.mean(axis=2)
    for c in corners.values():
        uv, _ = cam.project(c)
        border = (uv[0] + 0.05 * (uv[2] - uv[0])).round().astype(int)
        assert gray[border[1], border[0]] < 80
    plate = cam.project(MAZE_POSE[:3, 3] + [0.03, 0.03, 0])[0][0].round().astype(int)
    assert gray[plate[1], plate[0]] > 180


def test_random_pose():
    from crs_sim.puzzles import BASE_PLATE_THICKNESS, random_pose

    p = random_pose(0, table_z=0.02)
    assert 0.35 <= p[0, 3] <= 0.55 and -0.2 <= p[1, 3] <= 0.2
    assert p[2, 3] == pytest.approx(0.02 + BASE_PLATE_THICKNESS)
    assert np.allclose(random_pose(0, table_z=0.02), p)


def test_wire_through_hoop(puzzle_a):
    robot = make_robot()
    assert puzzle_a.wire_through_hoop(robot.fk(DESCENT[3]) @ HOOP_IN_FLANGE)
    assert not puzzle_a.wire_through_hoop(robot.fk(DESCENT[0]) @ HOOP_IN_FLANGE)
    assert not puzzle_a.wire_through_hoop(robot.fk(robot.q_home) @ HOOP_IN_FLANGE)
