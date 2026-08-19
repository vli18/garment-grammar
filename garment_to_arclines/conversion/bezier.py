"""GarmentCode's relative-coordinate Bezier curve evaluation.

rel_to_abs_2d matches pygarment/pattern/utils.py::rel_to_abs_2d exactly (no
y-flip correction needed here, unlike circle_arc.py: GarmentCode's rendering
flips both the vertex y and the control point y together for these edge
types, which is self-cancelling -- see circle_arc.py's docstring for why
circle edges are different).
"""

import numpy as np


def rel_to_abs_2d(start: np.ndarray, end: np.ndarray, rel_point: np.ndarray) -> np.ndarray:
    edge = np.asarray(end, dtype=np.float64) - np.asarray(start, dtype=np.float64)
    edge_perp = np.array([-edge[1], edge[0]])
    return np.asarray(start, dtype=np.float64) + rel_point[0] * edge + rel_point[1] * edge_perp


def sample_quadratic(start: np.ndarray, end: np.ndarray, control_rel: np.ndarray, t: float) -> np.ndarray:
    control = rel_to_abs_2d(start, end, control_rel)
    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    return (1 - t) ** 2 * start + 2 * (1 - t) * t * control + t**2 * end


def sample_cubic(start: np.ndarray, end: np.ndarray, control1_rel: np.ndarray, control2_rel: np.ndarray, t: float) -> np.ndarray:
    c1 = rel_to_abs_2d(start, end, control1_rel)
    c2 = rel_to_abs_2d(start, end, control2_rel)
    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    return (1 - t) ** 3 * start + 3 * (1 - t) ** 2 * t * c1 + 3 * (1 - t) * t**2 * c2 + t**3 * end
