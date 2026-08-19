"""Exact conversion of GarmentCode's `circle` edge curvature into D4D's
signed-sagitta Arc parameter `k`.

GarmentCode's `circle` curvature is literally an SVG circular-arc command:
params = [radius, large_arc_flag, right_flag], with sweep_flag = not right_flag
(see pygarment/pattern/core.py::_edge_as_curve upstream). D4D's Arc is the
same family of curve (a circle through two fixed endpoints) but parameterized
by a single signed sagitta k (perpendicular distance from the chord midpoint
to the arc, along perp = rotate(end-start, +90 deg)).

Derivation: solving D4D's o(k) = midpoint + (r(k)-k)*perp_unit, r(k) =
(k^2+c^2)/(2k), for k such that o(k) equals the true SVG-arc center gives a
quadratic in k with two roots, corresponding to the minor/major arc through
that center. Which root is which was confirmed numerically against
svgpathtools (GarmentCode's own rendering library) across all four
(large_arc, right) combinations -- see conversation history / commit for the
validation script.

Sign convention (important): SVG's sweep flag encodes an absolute
clockwise/counterclockwise handedness, defined for SVG's y-down image
coordinate frame. GarmentCode's actual rendering pipeline
(VisPattern._draw_a_panel) flips vertex y before feeding it to the SVG arc
builder to get there; for quadratic/cubic edges that flip is self-cancelling
because the control point's y is flipped too, but circle edges have no
control point to flip, so the flag is only ever meaningful in the flipped
frame. specification.json vertices are in the plain y-up cm frame, so using
the flags directly against un-flipped coordinates reverses which side the
arc bulges toward -- confirmed by comparing against GarmentCodeData's
rendered pattern.png (a real waistband panel should be a roughly
constant-width crescent; using the flags naively produced a
pinched/hourglass shape instead). The fix is a final sign flip below.
"""

import math

import numpy as np


def circle_edge_to_k(start: np.ndarray, end: np.ndarray, radius: float, large_arc: int, right: int) -> float:
    """
    start, end: (2,) endpoints, in the same coordinate frame/units as radius,
        in the plain y-up cm frame (as stored in specification.json).
    radius, large_arc, right: GarmentCode's circle curvature params.
    Returns: D4D's signed sagitta k, exact (no approximation).
    """
    d = np.asarray(end, dtype=np.float64) - np.asarray(start, dtype=np.float64)
    c = float(np.linalg.norm(d)) / 2
    h = math.sqrt(max(radius**2 - c**2, 0.0))
    sweep = not right
    sign = 1.0 if (bool(large_arc) != bool(sweep)) else -1.0
    if large_arc == 0:
        k = sign * (radius - h)
    else:
        k = -sign * (radius + h)
    return -k  # correct for the y-up vs. SVG y-down sweep convention mismatch
