# Jak pouzivat

Write src/main.py as normal real-robot code, starting with from ctu_crs import CRS97.

1. Connected to the real robot (on the lab PC, where ctu_crs is installed):

```python
python src/main.py
```

2. In the simulator, with the viewer opening in your browser:

```python
./sim.sh src/main.py
```

Options for the simulated scene go before `src/main.py`, for example `./sim.sh --puzzle C --seed 7 src/main.py`. `./sim.sh --help` lists them all.

# ...

# ...

# Andrlik slop

Standalone simulator of the **CRS97** robotic arm. It has the same interface as
the real robot's driver [`ctu_crs`](https://pypi.org/project/ctu-crs/), so code
written for the hardware runs on the simulator by changing one import:

```python
from ctu_crs import CRS97   # real robot
from crs_sim import CRS97   # simulator
```

The robot is rendered from its URDF + STL meshes in a [MeshCat](https://github.com/meshcat-dev/meshcat-python)
browser window.

## Installation

```bash
git clone <this repo> crs_sim && cd crs_sim
./setup_venv.sh            # creates .venv, installs crs_sim (editable) + pytest + STEP support
source .venv/bin/activate
```

`PYTHON=python3.11 ./setup_venv.sh` picks a different interpreter and
`EXTRAS=dev ./setup_venv.sh` skips the CAD/STEP support (~100 MB, OpenCASCADE). To use the
package from another environment (e.g. a conda env), run `pip install -e /path/to/crs_sim`.

Quick start:

```bash
./start.sh                  # interactive console with `robot` already created
./start.sh examples/demo.py # run a script inside the venv
./start.sh -m pytest        # tests (headless)
```

`start.sh` and `setup_venv.sh` unset `PYTHONPATH`, so a sourced ROS workspace
cannot leak its packages (and pytest plugins) into the venv. If you activate the
venv manually with ROS sourced, run `unset PYTHONPATH` first.

Dependencies: `numpy`, `pin` (pinocchio) and `meshcat`; the optional `step` extra
(`trimesh`, `cascadio`) loads CAD files. The real-robot packages (`ctu_crs`,
`pypylon`, serial port drivers) are **not** needed.

## Your solution: `src/main.py`

`src/main.py` is yours, it starts empty. Write it as plain real-robot code
(`from ctu_crs import CRS97`), it must not import `crs_sim`. The same file
runs on both:

```bash
python src/main.py                         # connected to the real robot (ctu_crs installed)
./sim.sh src/main.py                       # simulator with the viewer
./sim.sh --puzzle C --seed 7 src/main.py   # other puzzle / random placement
```

`./sim.sh` (= `python -m crs_sim.launch`) builds the scene, provides a
stand-in `ctu_crs` whose `CRS97` is the simulated robot in it, runs the
script and prints a verdict at the end (contacts, how far down the wire the
hoop got). Scene options go **before** the script, the script's own
arguments after it; `./sim.sh --help` lists them (`--puzzle A..E|none`,
`--seed` or `--x --y --yaw`, `--table-z`, `--rod ...`, `--lag`, `--noise`,
`--camera-width/height/fov`, `--speedup`, `--headless`, `--on-contact`,
`--contact-margin`).

## Usage

```python
import numpy as np
from crs_sim import CRS97

robot = CRS97()                  # opens the viewer in the browser
robot.initialize()               # hard home + soft home, like the real robot

q = robot.q_home + [0.3, 0.2, 0.0, 0.0, 0.0, 0.5]
robot.move_to_q(q)               # non-blocking, exactly like the real robot
robot.wait_for_motion_stop()
print(robot.get_q())

pose = robot.fk(q)                                   # 4x4 flange pose
sols = [s for s in robot.ik(pose) if robot.in_limits(s)]
```

## Examples

Run them with `./start.sh examples/<name>.py`, all take `--help`.

| Example                                                                      | What it shows                                                                                                                                                                                      |
| :--------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `demo.py`                                                                    | Reach a target pose with IK, draw frames                                                                                                                                                           |
| `puzzle_env.py [--puzzle C] [--seed 7] [--image f.png] [--trajectory q.txt]` | The puzzle task as an environment: any puzzle placed on the table, ground truth printed, camera frame saved, a joint trajectory replayed and judged (contacts, how far down the wire the hoop got) |
| `puzzle_a.py`                                                                | Puzzle A in the workspace, the hoop moved down the wire and back through fixed joint configurations                                                                                                |
| `hoop_on_rod.py [--lag 0.3] [--no-predict]`                                  | Track a moving rod in the lagged overhead camera image and slide the hoop onto it                                                                                                                  |
| `show_cad.py [files...]`                                                     | Show any STEP/STL/OBJ file in the workspace; without arguments all five puzzles                                                                                                                    |
| `draw_circle.py`                                                             | Cartesian path following: trace a circle with IK, draw the trail                                                                                                                                   |
| `joint_sweep.py`                                                             | Joint limits table and each joint swept towards its limits                                                                                                                                         |
| `real_or_sim.py [--real]`                                                    | The same control code on the simulator or the real robot                                                                                                                                           |

## Puzzles: a virtual replacement of the physical task

All five puzzles (A - E) of the ROB1 task are scene objects: the bent
10x10 mm wire plugged into the base plate with the ArUco markers 1 and 2
(`DICT_4X4_50`, 40 mm, raised black cells on a white plate, as printed).
Added to the robot, a puzzle is drawn in the viewer, rendered by the camera
(`grab_image`) and **every touch of the robot is detected**:

```python
import numpy as np
from crs_sim import CRS97
from crs_sim.puzzles import HOOP_IN_FLANGE, Puzzle, random_pose
from crs_sim.transforms import pose

robot = CRS97()                                   # on_contact="warn" | "raise" | "ignore"
puzzle = Puzzle("C", random_pose(seed=7))         # or Puzzle("C", pose([0.45, 0.05, 0.005], yaw=0.3))
robot.add_object(puzzle)
robot.initialize()

image = robot.grab_image()      # find the markers here, like on the real robot
...                             # your solution: move_to_q / wait_for_motion_stop
robot.contacts                  # [ContactEvent(time, q, [Contact(link, obstacle, point)])]
robot.clear_contacts()          # before the next attempt

# ground truth, to evaluate your perception
puzzle.pose                     # maze frame (wire origin on the plate top) in the base frame
puzzle.marker_corners()         # {1: 4x3, 2: 4x3} corners in cv2.aruco order
puzzle.marker_poses()           # {1: 4x4, 2: 4x4}
puzzle.wire_through_hoop(robot.fk(q) @ HOOP_IN_FLANGE)   # is the hoop on the wire
```

How contacts are checked: the robot's exact URDF meshes (hoop, adapter and
all links) against the puzzle's CAD meshes (wire and base plate), mesh-mesh
collision with coal (comes with pinocchio). Every executed motion is checked
in a background thread, densely enough that no robot point moves more than
`contact_resolution` (1 mm) between two checks; a motion interrupted by the
next `move_to_q` is checked only as far as the robot got. One touch
(however long) is one event, printed and marked red in the viewer.
`wait_for_motion_stop` returns once the motion is checked, so `robot.contacts`
is complete afterwards, also with `speedup=np.inf`. With
`on_contact="raise"`, the next `wait_for_motion_stop` / `move_to_q` raises
`crs_sim.ContactError`. `contact_margin` counts near misses as contacts too;
`robot.contact_checker.distance(q, links=["hoop"])` gives the clearance.
The moving rod is checked the same way.

`random_pose(seed, x=(0.35, 0.55), y=(-0.2, 0.2), yaw=(-pi, pi), table_z=0)`
draws a placement; `puzzle.pose = ...` moves a puzzle, `robot.remove_object`
removes it. `examples/puzzle_env.py --trajectory q.txt` replays a list of
joint configurations and prints a verdict.

The CAD models come from [pilcsimo/ROB1-2025-hw](https://github.com/pilcsimo/ROB1-2025-hw)
(GPL-3.0). They are **not** included here; `download_puzzles()` fetches them to
`~/.cache/crs_sim/` on first use (needs the `step` extra). `crs_sim.puzzles`
also has the hoop offset `HOOP_IN_FLANGE` (hoop center pose =
`robot.fk(q) @ HOOP_IN_FLANGE`, the wire passes along its z axis), the hoop
opening `HOOP_INNER_RADIUS` (30 mm) and the wire center line of puzzle A as an
example of a path (`maze_path("A")`, hoop poses in the maze frame).

To only draw CAD files (no camera, no contacts):

```python
from crs_sim.puzzles import show_puzzle
show_puzzle(robot.viz, "D", pose([0.45, 0.0, 0.0], yaw=0.5))
robot.viz.show_mesh("part", "my_part.step", pose([0.3, 0.2, 0.0]))
robot.viz.show_mesh("bracket", "bracket.stl", scale=0.001)       # STL in mm
```

`crs_sim.transforms` has the small 4x4 helpers used above (`rx`, `ry`, `rz`,
`translation`, `pose`, `interpolate`, `densify`).

## Fidelity to the real robot

|                                                  | Real `ctu_crs.CRS97`                                             | `crs_sim.CRS97`                                                                                                 |
| :----------------------------------------------- | :--------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------------------- |
| `fk`, `ik`, `fk_flange_pos`, DH params           | analytic DH model                                                | identical code, identical results (IK solutions are shuffled too)                                               |
| `q_min`, `q_max`, `q_home`                       | from IRC limits                                                  | identical                                                                                                       |
| `move_to_q`                                      | non-blocking, asserts `initialize()` was called and joint limits | same; the motion takes real time                                                                                |
| `get_q`, `in_motion`, `wait_for_motion_stop`     | read from the controller                                         | follow the simulated motion, `q` quantized to encoder (IRC) resolution                                          |
| `soft_home`, `hard_home`, `initialize(home=...)` |                                                                  | same behavior, hard home moves the joints in the real `B → AC → DEF` order                                      |
| `set_speed(_relative)`                           |                                                                  | changes the simulated joint speeds                                                                              |
| `set_acceleration(_relative)`                    |                                                                  | stored only (a smooth quintic profile is used)                                                                  |
| `gripper.*`                                      |                                                                  | tracks the commanded position, not rendered                                                                     |
| `grab_image`                                     | Basler camera                                                    | rendered image of the simulated overhead camera with a configurable lag (or `robot.camera_image` if you set it) |
| `release`, `reset_motors`, `close`               |                                                                  | no-ops                                                                                                          |

The URDF agrees with the DH model to < 1 mm. The rendered `adapter` link equals
the DH flange frame rotated by π about its y-axis.

## Simulator-only options

All of these are keyword arguments of `CRS97(...)`:

| Option               | Default  | Meaning                                                                                                                                                       |
| :------------------- | :------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `visualize`          | `True`   | `False` runs headless (no viewer, meshcat not needed; pinocchio is only imported once the camera renders)                                                     |
| `open_browser`       | `True`   | open the MeshCat page automatically                                                                                                                           |
| `speedup`            | `1.0`    | time scale of motions; `np.inf` makes them instantaneous                                                                                                      |
| `q0`                 | `q_home` | initial joint configuration                                                                                                                                   |
| `render_fps`         | `30`     | render rate during motions                                                                                                                                    |
| `camera`             | `True`   | overhead camera for `grab_image`: `True` = default, a `crs_sim.camera.Camera(...)` to configure it, `False` = none                                            |
| `on_contact`         | `"warn"` | touching an added object: `"warn"` prints it and marks it in the viewer, `"raise"` also raises `ContactError`, `"ignore"` only records it in `robot.contacts` |
| `contact_resolution` | `0.001`  | largest motion of any robot point between two contact checks [m]                                                                                              |
| `contact_margin`     | `0.0`    | report a contact already when closer than this [m]                                                                                                            |

## Camera and moving rod

`robot.camera` is a pinhole camera 1.6 m above the workspace looking straight
down (base +x points up in the image). `robot.grab_image()` renders the robot,
the hoop and every object added with `robot.add_object` into an 2448x2050 BGR
image, the way it looked `lag` seconds ago:

```python
import numpy as np
from crs_sim import CRS97
from crs_sim.camera import Camera, look_at, write_png
from crs_sim.rod import MovingRod, oscillating
from crs_sim.transforms import translation

robot = CRS97()                       # or CRS97(camera=Camera(width=1280, height=960, lag=0.2))
robot.camera.lag = 0.25               # input lag [s], change it any time
robot.camera.lag_jitter = 0.02        # random extra lag (std) [s]
robot.camera.fps = 30                 # exposures happen every 1/fps
robot.camera.noise = 2.0              # pixel noise (std) [intensity levels]

rod = MovingRod(oscillating(translation([0.45, 0, 0]), [0, 0.08, 0], period=10.0),
                radius=0.01, length=0.2)
robot.add_object(rod)                 # rendered by the camera, animated in the viewer

image = robot.grab_image()            # HxWx3 uint8, BGR (OpenCV order)
t_exp = robot.camera.last_exposure_time   # time.monotonic() the image shows
rod.pose(t_exp), rod.top()            # ground truth then / now
robot.q_at(t_exp)                     # robot joints then (last 10 s are kept)

K = robot.camera.K                    # intrinsics, robot.camera.pose = extrinsics (base frame)
uv, depth = robot.camera.project(points_nx3)
xyz = robot.camera.backproject(uv, z=0.2)  # pixels -> points on the plane z = 0.2
write_png("frame.png", image)
```

How the lag works: a frame exposed at `t_exp` (a multiple of `1/fps`) is
delivered at `t_exp + lag`, and `grab_image` returns the newest delivered
frame. Rendering takes ~0.1 s, so `grab_image` blocks about that long (like
waiting for a frame) and picks the exposure so that the image is exactly `lag`
old when it returns (plus up to one frame period); a lag shorter than the
rendering time is not reachable. Lag, `fps` and the rod trajectory are in
wall-clock seconds, `speedup` only scales robot motions.

The image is rendered by a small numpy point-splatting renderer (URDF meshes,
headlight shading, z-buffer, no shadows): good for color segmentation,
circle/ellipse fitting and back-projection, not photorealistic. The hoop is
blue (inner radius 30 mm), the rod red by default. Rod trajectories in
`crs_sim.rod`: `static`, `linear`, `oscillating`, `circular`, or any function
`t -> 4x4 pose` (t = seconds since the rod was created, `rod.restart()` resets
it). Any object with `surfaces_at(t)` returning `[(crs_sim.camera.Surface,
pose_4x4), ...]` can be added to the camera too.

## Drawing into the scene

`robot.viz` (a `crs_sim.Visualizer`) draws helper objects:

```python
robot.viz.show_points("path", points_nx3, color=0xffa500, radius=0.005)
robot.viz.show_frame("target", pose_4x4, scale=0.1)
robot.viz.show_box("table", [1.8, 0.8, 0.02], pose_4x4, opacity=0.3)
robot.viz.show_mesh("part", "part.step", pose_4x4)   # any CAD file, needs the step extra
robot.viz.show_cylinder("rod", radius, length, pose_4x4)  # axis = pose z, from its origin up
robot.viz.show_lines("lines", segments_nx2x3)
robot.viz.set_pose("rod", pose_4x4)                  # move an object already drawn
robot.viz.set_camera([1.2, -1.4, 1.0])
robot.viz.delete("path")      # or robot.viz.delete() for all helper objects
robot.viz.viewer              # raw meshcat.Visualizer
```

## License

The kinematics and robot parameters are taken from `ctu_crs` (BSD 3-Clause,
Copyright (c) 2023, CTURobotics). See `LICENSE-ctu_crs`.
