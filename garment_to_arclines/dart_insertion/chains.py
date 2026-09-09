"""Arc-length utilities for multi-primitive edge chains -- e.g. a curved
armhole made of several Arc/Line primitives after Bezier-fitting (see
synthetic_panel.py's curved=True panel, or any real GarmentCodeData edge)."""

from typing import Callable

import torch

from d4descent.objects.arclines import Arc, Line, split_arc

# NOT using d4descent's split_line: its midpoint formula is
# `t * (line.start + line.end)` instead of `line.start + t*(line.end -
# line.start)` -- only correct at t=0.5 by coincidence (e.g. t=0 returns
# the origin instead of the line's own start point). Confirmed by direct
# test, not assumed. split_arc is fine (proper parametric sampling); it's
# specifically split_line that's broken, so Lines are split by hand below.


def _split_line(line: Line, t: float) -> tuple[Line, Line]:
    point = line.start + t * (line.end - line.start)
    return Line(line.start, point), Line(point, line.end)


def primitive_length(prim: Line | Arc) -> torch.Tensor:
    if isinstance(prim, Arc):
        _o, r, theta0, theta1 = prim.compute_o_r_thetas()
        return r * (theta1 - theta0).abs()
    return (prim.end - prim.start).norm()


def chain_length(chain: list[Line | Arc]) -> torch.Tensor:
    total = torch.zeros(())
    for p in chain:
        total = total + primitive_length(p)
    return total


def locate_arc_length(chain: list[Line | Arc], target: torch.Tensor) -> tuple[int, float]:
    """Which primitive `target` arc-length distance (from chain[0].start)
    falls in, and the local t within it."""
    acc = torch.zeros(())
    for i, prim in enumerate(chain):
        length = primitive_length(prim)
        if i == len(chain) - 1 or acc + length >= target:
            local_t = float(((target - acc) / length).clamp(0.0, 1.0)) if length > 0 else 0.0
            return i, local_t
        acc = acc + length
    raise AssertionError("unreachable: loop always returns by the last-primitive case")


def _ensure_low_to_high(chain: list[Line | Arc]) -> tuple[list[Line | Arc], bool]:
    """Returns (walk, was_reversed): walk goes from whichever endpoint
    (chain[0].start or chain[-1].end) has the lower y, matching
    geometry.one_third_up's single-primitive convention."""
    start_pt, end_pt = chain[0].start, chain[-1].end
    if start_pt[1] <= end_pt[1]:
        return chain, False
    return [p.reverse() for p in reversed(chain)], True


def split_chain_by_arc_length(
    chain: list[Line | Arc], fraction: float
) -> tuple[torch.Tensor, list[Line | Arc], list[Line | Arc]]:
    """Split chain at `fraction` of its total arc-length, walking from
    whichever end has the lower y. Returns (point, high_side, low_side):
    both sub-chains in the ORIGINAL chain's order/orientation (high_side
    then low_side reconstructs the original chain exactly), meeting
    exactly at `point`. For the armhole (built V2 -> V3, V2 the higher
    point), high_side is the V2-to-hinge portion (stays with center-front)
    and low_side is the hinge-to-V3 portion (rotates with upper-side).
    """
    walk, was_reversed = _ensure_low_to_high(chain)
    total = chain_length(walk)
    target = total * fraction
    idx, local_t = locate_arc_length(walk, target)

    prim = walk[idx]
    if isinstance(prim, Arc):
        lo_half, hi_half = split_arc(prim, local_t)
    else:
        lo_half, hi_half = _split_line(prim, local_t)
    point = lo_half.end

    walk_lo = walk[:idx] + [lo_half]
    walk_hi = [hi_half] + walk[idx + 1 :]

    def to_original(sub_walk: list[Line | Arc]) -> list[Line | Arc]:
        return [p.reverse() for p in reversed(sub_walk)]

    if was_reversed:
        # walk went low->high but original chain goes high->low
        return point, to_original(walk_hi), to_original(walk_lo)
    else:
        return point, walk_lo, walk_hi


def sample_chain(chain: list[Line | Arc], n_per_arc: int = 20) -> list[torch.Tensor]:
    """Point list tracing chain -- each primitive's start, plus intermediate
    samples for arcs (excluding each primitive's own end, so consecutive
    primitives don't duplicate their shared joint). Does not include the
    very last point (chain[-1].end); treat the result as an open polyline
    or, if the chain is itself one edge of a closed loop, let the caller's
    convention (e.g. polygon_area's implicit wraparound) supply that last
    edge. For a closed boundary made of several such chains concatenated,
    call this once on the full concatenated list."""
    pts: list[torch.Tensor] = []
    for p in chain:
        if isinstance(p, Arc):
            ts = torch.linspace(0, 1, n_per_arc)
            pts.extend(p.sample(ts)[:-1])
        else:
            pts.append(p.start)
    return pts


def transform_chain(chain: list[Line | Arc], transform_fn: Callable[[torch.Tensor], torch.Tensor]) -> list[Line | Arc]:
    """Apply transform_fn (a point -> point rigid motion, e.g.
    geometry.rotate_about(_, hinge, theta) or lambda p: p + trans) to every
    primitive in chain. k is kept as-is for arcs -- verified in
    conversation that k is invariant under rotation/translation as long as
    both endpoints are transformed by the same rigid motion."""
    result: list[Line | Arc] = []
    for p in chain:
        new_start, new_end = transform_fn(p.start), transform_fn(p.end)
        if isinstance(p, Arc):
            result.append(Arc(new_start, new_end, p.k))
        else:
            result.append(Line(new_start, new_end))
    return result
