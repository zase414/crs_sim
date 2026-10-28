"""
Run unmodified real-robot code in the simulator.

    python -m crs_sim.launch [scene options] script.py [script arguments]
    ./sim.sh --puzzle C --seed 7 src/main.py

The script is written for the real robot (``from ctu_crs import CRS97`` or
``from ctu_mitsubishi import Rv6s``) and knows nothing about the simulator.
The launcher builds the scene from its own options, puts stand-in
``ctu_crs`` and ``ctu_mitsubishi`` modules in ``sys.modules`` whose robot
classes create the simulated robot with that scene added, and runs the
script as ``__main__`` with its own arguments. Whichever robot the script
creates is the one in the scene. When the script ends (also on an error or
Ctrl+C) the verdict is printed: every contact and how far down the wire the
hoop got.

Everything after the script path is passed to the script unchanged, put the
scene options before it.
"""
from __future__ import annotations

import argparse
import runpy
import sys
import types
from pathlib import Path

import numpy as np

from .camera import Camera
from .crs97 import CRS97 as SimCRS97
from .rv6s import Rv6s as SimRv6s
from .rv6s import Rv6sGripper as SimRv6sGripper
from .transforms import pose, translation


class Scene:
    """The simulated world the launched script's robot lives in."""
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.robot = None
        self.puzzle = None
        self.rod = None
        #: lowest hoop center height above the plate top with the wire
        #: through the hoop [m], None if the hoop never got onto the wire
        self.lowest_threaded = None

    def camera(self) -> Camera:
        a = self.args
        return Camera(width=a.camera_width, height=a.camera_height,
                      fov=np.deg2rad(a.camera_fov), lag=a.lag, noise=a.noise)

    def create_robot(self, tty_dev="/dev/mars", baudrate=19200, **kwargs):
        """Stand-in for ctu_crs.CRS97: the simulated robot in the scene.
        There is one robot, every call returns it."""
        return self._create(SimCRS97, tty_dev, baudrate)

    def create_rv6s(self, port="/dev/ttyUSB_Robot", baudrate=38400, debug=False):
        """Stand-in for ctu_mitsubishi.Rv6s, see create_robot."""
        return self._create(SimRv6s, port, baudrate, debug)

    def _create(self, cls, *args):
        if self.robot is not None:
            if not isinstance(self.robot, cls):
                raise RuntimeError(
                    f"The simulator has one robot, the script already created "
                    f"a {type(self.robot).__name__}, it cannot also create a "
                    f"{cls.__name__}.")
            return self.robot
        a = self.args
        robot = cls(*args, visualize=not a.headless, speedup=a.speedup,
                    camera=self.camera(), on_contact=a.on_contact,
                    contact_margin=a.contact_margin)
        self.robot = robot
        if robot.viz is not None:
            robot.viz.show_box("table", [1.2, 1.2, 0.02],
                               translation([0.3, 0, a.table_z - 0.01]),
                               color=0x8b6b4a)
        if a.puzzle != "none":
            self._add_puzzle(robot)
        if a.rod != "none":
            self._add_rod(robot)
        return robot

    def _add_puzzle(self, robot) -> None:
        from .puzzles import BASE_PLATE_THICKNESS, Puzzle, random_pose

        a = self.args
        if a.seed is not None:
            maze_pose = random_pose(a.seed, table_z=a.table_z)
        else:
            maze_pose = pose([a.x, a.y, a.table_z + BASE_PLATE_THICKNESS], a.yaw)
        self.puzzle = Puzzle(a.puzzle, maze_pose)
        robot.add_object(self.puzzle)
        to_maze = np.linalg.inv(maze_pose)

        def track_depth(q, t):
            hoop = robot.fk(q) @ robot.hoop_in_flange
            if self.puzzle.wire_through_hoop(hoop):
                h = (to_maze @ hoop)[2, 3]
                if self.lowest_threaded is None or h < self.lowest_threaded:
                    self.lowest_threaded = h
        robot.add_monitor(track_depth)

    def _add_rod(self, robot) -> None:
        from .rod import MovingRod, circular, oscillating, static

        a = self.args
        center = translation([a.rod_x, a.rod_y, a.table_z])
        trajectory = {"static": lambda: static(center),
                      "oscillating": lambda: oscillating(
                          center, [0, a.rod_amplitude, 0], a.rod_period),
                      "circular": lambda: circular(
                          center, a.rod_amplitude, a.rod_period)}[a.rod]()
        self.rod = MovingRod(trajectory)
        robot.add_object(self.rod)

    def describe(self) -> str:
        lines = [f"robot {type(self.robot).__name__}"] if self.robot else []
        if self.puzzle is not None:
            p = self.puzzle.pose
            yaw = np.arctan2(p[1, 0], p[0, 0])
            lines.append(f"puzzle {self.puzzle.name} at x={p[0, 3]:.3f} "
                         f"y={p[1, 3]:.3f} z={p[2, 3]:.4f} m, yaw={yaw:.3f} rad")
        if self.rod is not None:
            lines.append(f"rod: {self.args.rod} around "
                         f"x={self.args.rod_x:.3f} y={self.args.rod_y:.3f} m")
        return "\n".join(lines) or "empty scene"

    def verdict(self) -> str:
        robot = self.robot
        if robot is None:
            return ("The script never created a robot "
                    "(ctu_crs.CRS97 or ctu_mitsubishi.Rv6s).")
        # the contact check may still be catching up with the last motion
        robot.settle()
        lines = ["Scene: " + self.describe().replace("\n", ", ")]
        lines += [f"  {event}" for event in robot.contacts]
        lines.append(f"  {len(robot.contacts)} contact(s)")
        if self.puzzle is not None:
            if self.lowest_threaded is None:
                lines.append("  the hoop never got onto the wire")
            else:
                lines.append(f"  hoop threaded down to "
                             f"{1000 * self.lowest_threaded:.0f} mm above the base plate")
        return "\n".join(lines)


def fake_ctu_crs(scene: Scene) -> types.ModuleType:
    """A ``ctu_crs`` module whose CRS97 is the simulated robot of the scene."""
    module = types.ModuleType("ctu_crs")
    module.__doc__ = "crs_sim stand-in for ctu_crs (only CRS97 is simulated)"
    module.CRS97 = scene.create_robot

    def missing(name):
        raise AttributeError(f"ctu_crs.{name} is not simulated by crs_sim, "
                             "only CRS97 is")
    module.__getattr__ = missing
    return module


def fake_ctu_mitsubishi(scene: Scene) -> types.ModuleType:
    """A ``ctu_mitsubishi`` module whose Rv6s is the simulated robot of the scene."""
    module = types.ModuleType("ctu_mitsubishi")
    module.__doc__ = "crs_sim stand-in for ctu_mitsubishi"
    module.Rv6s = scene.create_rv6s
    module.Rv6sGripper = SimRv6sGripper

    def missing(name):
        raise AttributeError(f"ctu_mitsubishi.{name} is not simulated by crs_sim, "
                             "only Rv6s and Rv6sGripper are")
    module.__getattr__ = missing
    return module


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m crs_sim.launch",
        description="Run real-robot code (from ctu_crs import CRS97 / "
                    "from ctu_mitsubishi import Rv6s) in the simulator.",
        epilog="Scene options go before the script, the script's own after it.")
    scene = parser.add_argument_group("scene")
    scene.add_argument("--puzzle", default="A", choices=[*"ABCDE", "none"])
    scene.add_argument("--seed", type=int,
                       help="random puzzle placement (default: --x/--y/--yaw)")
    scene.add_argument("--x", type=float, default=0.45, help="puzzle position [m]")
    scene.add_argument("--y", type=float, default=0.0, help="puzzle position [m]")
    scene.add_argument("--yaw", type=float, default=0.0, help="puzzle rotation [rad]")
    scene.add_argument("--table-z", type=float, default=0.0,
                       help="table top height in the robot base frame [m]")
    scene.add_argument("--rod", default="none",
                       choices=["none", "static", "oscillating", "circular"])
    scene.add_argument("--rod-x", type=float, default=0.45, help="rod center [m]")
    scene.add_argument("--rod-y", type=float, default=0.0, help="rod center [m]")
    scene.add_argument("--rod-amplitude", type=float, default=0.08,
                       help="oscillation amplitude / circle radius [m]")
    scene.add_argument("--rod-period", type=float, default=10.0, help="[s]")

    cam = parser.add_argument_group("camera")
    cam.add_argument("--lag", type=float, default=0.1, help="input lag [s]")
    cam.add_argument("--noise", type=float, default=2.0, help="pixel noise (std)")
    cam.add_argument("--camera-width", type=int, default=2448)
    cam.add_argument("--camera-height", type=int, default=2050)
    cam.add_argument("--camera-fov", type=float, default=60.0,
                     help="horizontal field of view [deg]")

    sim = parser.add_argument_group("simulation")
    sim.add_argument("--speedup", type=float, default=1.0,
                     help="motion time scale, inf = instant")
    sim.add_argument("--headless", action="store_true", help="no viewer")
    sim.add_argument("--on-contact", default="warn", choices=["warn", "raise", "ignore"])
    sim.add_argument("--contact-margin", type=float, default=0.0, help="[m]")
    sim.add_argument("--no-wait", action="store_true",
                     help="exit right after the script, do not keep the viewer open")

    parser.add_argument("script", help="the real-robot script to run")
    parser.add_argument("script_args", nargs=argparse.REMAINDER,
                        help="arguments passed to the script")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    scene = Scene(args)
    script = Path(args.script).resolve()
    sys.modules["ctu_crs"] = fake_ctu_crs(scene)
    sys.modules["ctu_mitsubishi"] = fake_ctu_mitsubishi(scene)
    sys.argv = [str(script), *args.script_args]
    sys.path.insert(0, str(script.parent))   # as `python script.py` does

    code = 0
    try:
        runpy.run_path(str(script), run_name="__main__")
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    except KeyboardInterrupt:
        print("\n[crs_sim] interrupted")
        code = 130
    finally:
        print("\n[crs_sim] Verdict:\n" + scene.verdict())

    robot = scene.robot
    if (robot is not None and robot.viz is not None and not args.no_wait
            and sys.stdin.isatty()):
        input("[crs_sim] Press enter to close the simulator.")
    return code


if __name__ == "__main__":
    sys.exit(main())
