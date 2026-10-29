"""
Key vectors the hoop center has to pass through for puzzles A - D.

Saved as src/utils/key_vectors/puzzle_{a,b,c,d}.npy, each an Nx6 array of
rows [x, y, z, dx, dy, dz]:
  - x, y, z    hoop center on the wire center line, maze frame [m]
               (origin where the wire stands on the base plate top, z up)
  - dx, dy, dz unit wire tangent pointing toward the base plate, i.e. the
               direction the hoop travels when threading it on
Rows go from the free end of the wire down to the base. A sharp bend appears
twice: the same point with the incoming and with the outgoing direction.

The center lines below were measured from the STEP models of
https://github.com/pilcsimo/ROB1-2025-hw/tree/main/puzzles (10x10 mm wire),
run this file to regenerate the arrays and check them against the meshes:

    .venv/bin/python src/utils/make_key_vectors.py
"""
from pathlib import Path

import numpy as np

#: Lowest hoop center above the base plate top [m]
BOTTOM_Z = 0.01
#: Angular step when sampling the bend of puzzle D [rad]
ARC_STEP = np.deg2rad(15)

OUT_DIR = Path(__file__).parent / "key_vectors"


def arc(center, radius, a0, a1, step=ARC_STEP):
    """Points of an arc in the maze xz plane, angles from the +x axis, with
    their tangents in the direction of the arc: [x, y, z, tx, ty, tz] rows."""
    n = max(1, int(np.ceil(abs(a1 - a0) / step)))
    a = np.linspace(a0, a1, n + 1)
    s = np.sign(a1 - a0)
    return [[center[0] + radius * np.cos(t), 0.0, center[1] + radius * np.sin(t),
             -s * np.sin(t), 0.0, s * np.cos(t)] for t in a]


# Wire center line from the base (z = 0) to the free end, as polylines.
# Each entry is a list of "pieces"; consecutive pieces share their end point,
# a new piece starts a sharp bend, points within a piece are a smooth curve.
# A point is [x, y, z] or [x, y, z, tx, ty, tz] with its exact tangent (toward
# the free end), otherwise the tangent comes from the neighbouring points.
CENTER_LINES = {
    "A": [[[0, 0, 0], [0, 0, 0.2]]],
    "B": [[[0, 0, 0], [0, 0, 0.08]],
          [[0, 0, 0.08], [0.07, 0, 0.15]],
          [[0.07, 0, 0.15], [0.07, 0, 0.2]]],
    "C": [[[0, 0, 0], [0, 0, 0.05]],
          [[0, 0, 0.05], [-0.05, 0, 0.1]],
          [[-0.05, 0, 0.1], [-0.05, -0.05, 0.15]],
          [[-0.05, -0.05, 0.15], [-0.05, -0.05, 0.2]]],
    "D": [[[0, 0, 0], [0, 0, 0.03]],
          [[0, 0, 0.03], [-0.015, 0, 0.045]],
          # vertical, R50 bend to horizontal, horizontal to the free end
          [[-0.015, 0, 0.045]]
          + arc([0.035, 0.105], 0.05, np.pi, np.pi / 2)
          + [[0.085, 0, 0.155]]],
}


def key_vectors(name: str) -> np.ndarray:
    """Nx6 key vectors of a puzzle, from the free end down to the base."""
    rows = []
    for piece in CENTER_LINES[name]:
        p = np.array([pt[:3] for pt in piece], dtype=float)
        # clip the bottom so the hoop stays above the base plate
        if p[0, 2] < BOTTOM_Z:
            p[0] += (p[1] - p[0]) * (BOTTOM_Z - p[0, 2]) / (p[1, 2] - p[0, 2])
        seg = np.diff(p, axis=0)
        seg /= np.linalg.norm(seg, axis=1, keepdims=True)
        # segment direction at the ends, the average of the neighbouring
        # segments inside a smooth curve, unless given
        tangent = np.vstack([seg[:1], seg[:-1] + seg[1:], seg[-1:]])
        for i, pt in enumerate(piece):
            if len(pt) == 6:
                tangent[i] = pt[3:]
        tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
        # flip to point toward the base
        rows.extend(np.hstack([p, -tangent]))
    return np.array(rows[::-1])


def check(name: str, vectors: np.ndarray) -> float:
    """Largest distance of the wire mesh vertices from the center line [m],
    0.00707 for a 10x10 mm wire (more at the mitred corners)."""
    import sys
    sys.path.insert(0, str(Path(__file__).parents[2]))
    from crs_sim.puzzles import download_puzzles
    from crs_sim.visualizer import load_mesh

    mesh = load_mesh(download_puzzles(names=[name])[name])
    v = np.asarray(mesh.vertices)
    v = v[v[:, 2] > BOTTOM_Z]
    pts = vectors[:, :3]
    dist = np.full(len(v), np.inf)
    for a, b in zip(pts[:-1], pts[1:]):
        ab = b - a
        if ab @ ab == 0:
            continue
        t = np.clip((v - a) @ ab / (ab @ ab), 0, 1)
        dist = np.minimum(dist, np.linalg.norm(v - (a + t[:, None] * ab), axis=1))
    return dist.max()


def main():
    OUT_DIR.mkdir(exist_ok=True)
    np.set_printoptions(precision=4, suppress=True)
    for name in CENTER_LINES:
        vectors = key_vectors(name)
        path = OUT_DIR / f"puzzle_{name.lower()}.npy"
        np.save(path, vectors)
        print(f"{path}: {len(vectors)} key vectors, "
              f"max wire offset {1000 * check(name, vectors):.2f} mm")
        print(vectors)


if __name__ == "__main__":
    main()
