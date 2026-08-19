"""Independent ground-truth boundary sampling for GarmentCodeData panels,
used only for verification (overlay/comparison plots) -- deliberately does
NOT reuse loader.py's conversion math, so the check isn't circular.

Uses svgpathtools (the same curve library GarmentCode itself uses to render
its patterns) to sample the true curve for each edge type.
"""

import json

import numpy as np
import svgpathtools as svgpath


def _rel_to_abs_2d(start: np.ndarray, end: np.ndarray, rel_point: np.ndarray) -> np.ndarray:
    """Independent copy of GarmentCode's relative->absolute control point
    formula (see bezier.py) -- duplicated rather than imported so this stays
    a genuinely separate check from loader.py's own fitting code."""
    edge = end - start
    edge_perp = np.array([-edge[1], edge[0]])
    return start + rel_point[0] * edge + rel_point[1] * edge_perp


def _sample_edge(start: np.ndarray, end: np.ndarray, curvature: dict | None, n: int) -> np.ndarray:
    """Returns (n, 2) points from start to end (inclusive), tracing the true edge."""
    if curvature is None:
        ts = np.linspace(0, 1, n)
        return start[None, :] + ts[:, None] * (end - start)[None, :]

    if curvature["type"] == "circle":
        radius, large_arc, right = curvature["params"]
        # SVG's sweep flag is defined for a y-down frame; GarmentCode's own
        # renderer flips vertex y before applying it (see circle_arc.py for
        # the full explanation). Do the same here -- sample in the flipped
        # frame, then flip back -- so this stays a genuinely independent
        # check rather than reusing circle_arc.py's sign fix directly.
        arc = svgpath.Arc(
            complex(start[0], -start[1]),
            radius + 1j * radius,
            rotation=0,
            large_arc=large_arc,
            sweep=not right,
            end=complex(end[0], -end[1]),
        )
        ts = np.linspace(0, 1, n)
        pts = [arc.point(t) for t in ts]
        return np.array([[p.real, -p.imag] for p in pts])

    if curvature["type"] == "quadratic":
        control = _rel_to_abs_2d(start, end, np.array(curvature["params"][0], dtype=np.float64))
        curve = svgpath.QuadraticBezier(complex(*start), complex(*control), complex(*end))
        ts = np.linspace(0, 1, n)
        pts = [curve.point(t) for t in ts]
        return np.array([[p.real, p.imag] for p in pts])

    if curvature["type"] == "cubic":
        c1 = _rel_to_abs_2d(start, end, np.array(curvature["params"][0], dtype=np.float64))
        c2 = _rel_to_abs_2d(start, end, np.array(curvature["params"][1], dtype=np.float64))
        curve = svgpath.CubicBezier(complex(*start), complex(*c1), complex(*c2), complex(*end))
        ts = np.linspace(0, 1, n)
        pts = [curve.point(t) for t in ts]
        return np.array([[p.real, p.imag] for p in pts])

    raise NotImplementedError(f"Ground-truth sampling for curvature type '{curvature['type']}' not implemented yet.")


def sample_panel_boundary(spec_path: str, panel_name: str, n_per_edge: int = 32) -> np.ndarray:
    """Dense polyline tracing the true panel boundary (original cm coordinates),
    honoring each edge's actual curvature (not just straight vertex-to-vertex)."""
    with open(spec_path) as f:
        spec = json.load(f)

    panel = spec["pattern"]["panels"][panel_name]
    vertices = np.array(panel["vertices"], dtype=np.float64)
    edges = panel["edges"]

    points = []
    for edge in edges:
        start_idx, end_idx = edge["endpoints"]
        seg = _sample_edge(vertices[start_idx], vertices[end_idx], edge.get("curvature"), n_per_edge)
        points.append(seg[:-1])  # drop last point; next edge's first point picks it back up
    return np.concatenate(points, axis=0)
