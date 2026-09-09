"""Classifying panel points into the three dart-insertion regions by
angular sector from the apex."""

from enum import Enum

import torch


class Region(Enum):
    CENTER_FRONT = "center_front"
    UPPER_SIDE = "upper_side"
    LOWER_SIDE = "lower_side"


def _angle_from(apex: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    d = p - apex
    return torch.atan2(d[1], d[0])


def classify_point(
    point: torch.Tensor,
    apex: torch.Tensor,
    hinge: torch.Tensor,
    dart_point: torch.Tensor,
    hem_foot: torch.Tensor,
    eps: float = 1e-4,
) -> Region:
    """Which of the 3 angular sectors (measured from apex, going CCW)
    `point` falls in.

    Going CCW from line 2 (apex->hinge): first the upper-side sector (up to
    line 3 / apex->dart_point), then the lower-side sector (up to line 1 /
    apex->hem_foot), then the center-front sector (wrapping back to line 2).
    This ordering matches a CCW-wound panel boundary; if it doesn't hold
    (e.g. the rays aren't in that rotational order) something upstream is
    wrong, so this asserts rather than silently misclassifying.

    A point within `eps` radians of a ray is ambiguous -- it's on the cut
    itself. Rather than silently picking a side, this raises so the caller
    has to look at it (none of the synthetic panel's vertices hit this).
    """
    two_pi = 2 * torch.pi

    def rel(p: torch.Tensor) -> torch.Tensor:
        return (_angle_from(apex, p) - _angle_from(apex, hinge)) % two_pi

    r_dart = rel(dart_point)
    r_hem = rel(hem_foot)
    assert 0 < r_dart < r_hem < two_pi, (
        f"rays out of expected CCW order (hinge, dart_point, hem_foot): "
        f"r_dart={r_dart:.4f}, r_hem={r_hem:.4f} -- classify_point's sector logic assumes this order"
    )

    r_p = rel(point)
    for boundary_name, boundary_val in [("hinge", torch.tensor(0.0)), ("dart_point", r_dart), ("hem_foot", r_hem)]:
        if abs(float(r_p - boundary_val)) < eps or abs(float(r_p - boundary_val) - two_pi) < eps:
            raise ValueError(f"point lies within eps of ray to {boundary_name} (r_p={r_p:.6f}) -- ambiguous region")

    if r_p < r_dart:
        return Region.UPPER_SIDE
    elif r_p < r_hem:
        return Region.LOWER_SIDE
    else:
        return Region.CENTER_FRONT
