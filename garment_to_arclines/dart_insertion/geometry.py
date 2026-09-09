"""Small geometric helpers shared across dart-insertion phases."""

import torch

from d4descent.objects.arclines import Line


def fraction_along(line: Line, t: float) -> torch.Tensor:
    """Point at fraction t along the line, from start (t=0) to end (t=1)."""
    return line.start + t * (line.end - line.start)


def one_third_up(line: Line) -> torch.Tensor:
    """Point 1/3 of the way from the line's lower-y endpoint toward its
    higher-y endpoint -- used for the armhole hinge."""
    lo, hi = (line.start, line.end) if line.start[1] <= line.end[1] else (line.end, line.start)
    return lo + (hi - lo) / 3


def perpendicular_foot(point: torch.Tensor, line: Line) -> torch.Tensor:
    """Foot of the perpendicular dropped from point onto line's infinite
    extension (not clamped to the segment)."""
    d = line.end - line.start
    t = ((point - line.start) * d).sum() / (d * d).sum()
    return line.start + t * d


def rotate_about(point: torch.Tensor, pivot: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """Rotate point by theta (radians, CCW positive) about pivot."""
    v = point - pivot
    cos_t, sin_t = torch.cos(theta), torch.sin(theta)
    rotated = torch.stack([cos_t * v[0] - sin_t * v[1], sin_t * v[0] + cos_t * v[1]])
    return pivot + rotated


def line_intersection(p1: torch.Tensor, d1: torch.Tensor, p2: torch.Tensor, d2: torch.Tensor) -> torch.Tensor:
    """Intersection of infinite line (p1 + s*d1) with infinite line (p2 + u*d2)."""
    a = torch.stack([d1, -d2], dim=1)
    b = p2 - p1
    s, _u = torch.linalg.solve(a, b)
    return p1 + s * d1


def polygon_area(points: list[torch.Tensor]) -> torch.Tensor:
    """Signed area via the shoelace formula -- the standard measure of "the
    same 2D shape" for a filled region, insensitive to zero-area degenerate
    retraces (redundant collinear/there-and-back vertices)."""
    total = torch.zeros(())
    n = len(points)
    for i in range(n):
        x1, y1 = points[i][0], points[i][1]
        x2, y2 = points[(i + 1) % n][0], points[(i + 1) % n][1]
        total = total + (x1 * y2 - x2 * y1)
    return total / 2
