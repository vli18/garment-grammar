"""The dart-insertion equation, as standalone functions -- no region
splitting or rigid-motion application yet (see synthetic_panel.py /
visualize_phase_a.py for the panel and ray geometry these operate on).

theta = arcsin(sin(alpha) - t/a) - alpha

alpha: angle at the apex between line 1 (apex->hem) and line 2
       (apex->armhole hinge)
a:     length of line 2 (apex->hinge)
t:     spread parameter (input)
"""

import torch


def angle_at_apex(apex: torch.Tensor, p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
    """Acute angle (in [0, pi/2]) between ray apex->p1 and ray apex->p2.

    The raw ray-to-ray vertex angle is obtuse for line1 (apex->hem) vs
    line2 (apex->hinge) -- the hinge sits up toward the shoulder while line
    1 points down -- and dart_angle's arcsin(sin(alpha)) only round-trips
    to alpha for alpha in [-pi/2, pi/2]. Feeding it the raw obtuse angle
    puts tan(alpha + theta/2) exactly on a tan(90deg) singularity at the
    t=0 zero-state, which can't be right, so this returns the acute
    angle between the two lines (flipping to the supplement when the raw
    vertex angle is obtuse), matching the usual "angle between two lines"
    convention.
    """
    v1 = p1 - apex
    v2 = p2 - apex
    cos_angle = (v1 * v2).sum() / (v1.norm() * v2.norm())
    angle = torch.acos(cos_angle.clamp(-1.0, 1.0))
    return torch.pi - angle if angle > torch.pi / 2 else angle


def dart_angle(alpha: torch.Tensor, a: torch.Tensor, t: float) -> torch.Tensor:
    """theta = arcsin(sin(alpha) - t/a) - alpha.

    Note: arcsin(sin(alpha)) == alpha only holds for alpha in [-pi/2, pi/2].
    If alpha (apex->hem vs apex->hinge) comes out obtuse for a given panel,
    theta(t=0) will NOT come out to exactly 0 -- that's a real signal about
    the geometry, not a bug in this function, so it's deliberately not
    special-cased here. See phase_b_check.py's printed output.
    """
    return torch.asin(torch.sin(alpha) - t / a) - alpha


def outward_unit(down_unit: torch.Tensor, toward_side_seam: torch.Tensor) -> torch.Tensor:
    """Unit vector perpendicular to down_unit, oriented toward the side
    seam. toward_side_seam (e.g. dart_point - apex) is used only to pick
    the sign of the perpendicular -- its own magnitude/exact direction
    don't matter."""
    perp = torch.stack([-down_unit[1], down_unit[0]])
    sign = torch.sign((perp * toward_side_seam).sum())
    return perp * sign


def lower_region_translation(
    apex: torch.Tensor,
    hem_foot: torch.Tensor,
    dart_point: torch.Tensor,
    alpha: torch.Tensor,
    theta: torch.Tensor,
    t: float,
) -> torch.Tensor:
    """translation = t*outward_unit + t*tan(alpha + theta/2)*down_unit,
    with outward_unit/down_unit derived from the panel's own geometry, not
    fixed world axes (outward flips sign between left/right torso panels)."""
    down = hem_foot - apex
    down_unit = down / down.norm()
    out_unit = outward_unit(down_unit, dart_point - apex)
    return t * out_unit + t * torch.tan(alpha + theta / 2) * down_unit
