"""Generic adaptive fitting of an arbitrary parametric 2D curve into a chain
of D4D Line/Arc primitives, within a tolerance.

Not Bezier-specific: works for any sample_fn(t) -> (2,) point, t in [0, 1].
Reused for both quadratic and cubic GarmentCode edges.
"""

from typing import Callable

import numpy as np
import torch

from d4descent.objects.arclines import Arc, Line, merge_arc

CurveFn = Callable[[float], np.ndarray]


def fit_curve_to_arcs(
    sample_fn: CurveFn,
    start_tensor: torch.Tensor,
    end_tensor: torch.Tensor,
    tol: float,
    t0: float = 0.0,
    t1: float = 1.0,
    max_depth: int = 12,
    n_check: int = 9,
) -> list[Line | Arc]:
    """
    Fits sample_fn restricted to global parameter range [t0, t1] (whose
    endpoints are start_tensor/end_tensor -- reused, not recreated, so the
    result stays connected by shared tensor identity to whatever start_tensor/
    end_tensor already connect to) with a single Arc if it is within tol of
    the true curve; otherwise splits at the midpoint and recurses.

    tol is in the same units as sample_fn's output.
    """
    tm = (t0 + t1) / 2
    pm = sample_fn(tm)
    pm_tensor = torch.tensor(pm, dtype=torch.float32)

    line1 = Line(start_tensor, pm_tensor)
    line2 = Line(pm_tensor, end_tensor)
    arc, _ = merge_arc(line1, line2)  # passes exactly through start, pm, end

    ts_check = np.linspace(t0, t1, n_check)
    max_dev = 0.0
    for t in ts_check:
        true_pt = sample_fn(t)
        local_t = (t - t0) / (t1 - t0)
        arc_pt = arc.sample(torch.tensor(local_t, dtype=torch.float32)).numpy()
        max_dev = max(max_dev, float(np.linalg.norm(true_pt - arc_pt)))

    if max_dev <= tol or max_depth <= 0:
        if max_depth <= 0 and max_dev > tol:
            print(f"fit_curve_to_arcs: max_depth reached with deviation {max_dev:.4g} > tol {tol:.4g}")
        return [arc]

    left = fit_curve_to_arcs(sample_fn, start_tensor, pm_tensor, tol, t0, tm, max_depth - 1, n_check)
    right = fit_curve_to_arcs(sample_fn, pm_tensor, end_tensor, tol, tm, t1, max_depth - 1, n_check)
    return left + right
