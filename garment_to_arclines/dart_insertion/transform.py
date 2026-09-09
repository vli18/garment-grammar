"""Phase C/D: the actual dart-insertion transform -- classify vertices into
the 3 regions, apply the 3 rigid motions, insert the dart notch, and
assemble the new panel boundary. Works uniformly on straight (curved=False)
and curved (curved=True) panels: the armhole is always handled as a chain
(split_chain_by_arc_length), which reduces to the old single-Line behavior
automatically when the chain happens to have exactly one primitive.

Line 1 (apex->hem) and line 2 (apex->hinge) are purely constructional --
they never appear as edges in the final boundary. hinge is an exact,
zero-gap transition point (it's rotate-about-hinge's own pivot, so it's
invariant under both the identity and the rotation). hem_foot is NOT
invariant -- center-front's copy stays put (identity) while lower-side's
copy moves by the translation -- so instead of stopping at the old
hem_foot, the new (translated) hem is extended along its own direction
until it meets the center-front line, and that intersection replaces the
old V0/hem_foot corner. Center front lengthens (or shortens) by however
much that takes.

apex' = rotate_about(apex, hinge, theta) -- equivalently apex + trans, the
two are exactly equal (both are "apex", just reached via the upper-side
region's rotation vs. the lower-side region's translation) -- is the exact,
zero-gap point the two dart legs converge to if extended all the way in.
It's stored as `apex_prime` and used internally (it's what the theta-angle
identity and undo_dart rely on), but it is NOT what's actually drawn.

The drawn `tip` is apex' pulled back a bit, per real tailoring practice
(dart legs are drawn toward the apex but stopped short of it, ~1" in
practice -- fitted garments vary, but this codebase doesn't model fit yet).
The pullback distance is min(1 inch in normalized units, the actual dart
gap = |rotated_dart_point - translated_dart_point|) -- capped at a fixed
target, but scaled down for a barely-open dart so a small dart doesn't get
a disproportionately deep notch, and importantly exactly 0 when the gap is
0 (t=0), so the boundary's *topology* (same primitive count/connectivity
at every t -- required for this to stay usable with D4D's gradient-based
optimization, matching gradient_sanity.py's whole premise) never has to
change, only degenerate smoothly. The one rough edge: min() has a kink
right where the gap crosses 1 inch -- continuous but not smooth there.
Flagged as something to soften later if it matters for optimization.
"""

from dataclasses import dataclass

import numpy as np
import torch

from conversion.loader import PanelData
from d4descent.objects.arclines import Arc, Line, Shape

from .chains import chain_length, split_chain_by_arc_length, transform_chain
from .equations import angle_at_apex, dart_angle, lower_region_translation
from .geometry import fraction_along, line_intersection, perpendicular_foot, rotate_about
from .regions import Region, classify_point
from .synthetic_panel import edge_by_label, edges_by_label

INCH_TO_CM = 2.54


def inches_to_t(panel: PanelData, inches: float) -> float:
    """Convert a real spread (inches, at line 1) into the normalized t
    units insert_dart takes -- same conversion used for tip_setback."""
    return inches * INCH_TO_CM * panel.transform.scale


@dataclass
class DartResult:
    boundary: list[Line | Arc]  # full assembled panel boundary, closed loop (last.end == first.start)
    apex: torch.Tensor
    hinge: torch.Tensor
    dart_point: torch.Tensor
    hem_foot: torch.Tensor
    apex_prime: torch.Tensor  # exact convergence point; see module docstring
    tip: torch.Tensor  # drawn point: apex pulled back toward the dart opening
    v0_original: torch.Tensor
    rotated_V3: torch.Tensor
    rotated_dart_point: torch.Tensor
    translated_dart_point: torch.Tensor
    translated_V4: torch.Tensor
    translated_hem_foot: torch.Tensor
    new_corner: torch.Tensor
    alpha: torch.Tensor
    theta: torch.Tensor
    t: float
    trans: torch.Tensor  # the lower-side region's translation vector
    cf_extension: float  # signed: new CF corner vs. original V0, along the CF edge direction (+lengthens)
    armhole_cf_side: list[Line | Arc]  # V2 -> hinge, unchanged (identity)
    armhole_upper_side_rotated: list[Line | Arc]  # hinge -> V3, AFTER rotation about hinge
    collar_chain: list[Line | Arc]  # V1 -> V1b, unchanged (identity)

    def to_shape(self) -> Shape:
        return Shape(self.boundary)


def insert_dart(
    panel: PanelData,
    apex_raw: tuple[float, float],
    side_seam_t: float,
    t: float,
    tip_setback_inches: float = 1.0,
) -> DartResult:
    apex = torch.tensor(panel.transform.apply(np.array(apex_raw)), dtype=torch.float32)

    cf_edge = edge_by_label(panel, "center_front")
    collar_chain = edges_by_label(panel, "collar")
    shoulder_edge = edge_by_label(panel, "shoulder")
    armhole_chain = edges_by_label(panel, "armhole")
    side_seam = edge_by_label(panel, "side_seam")
    hem = edge_by_label(panel, "hem")

    V0, V1 = cf_edge.start, cf_edge.end
    V1b = collar_chain[-1].end
    V2 = shoulder_edge.end
    V3 = armhole_chain[-1].end
    V4 = side_seam.end

    # hinge: 1/3 of the way up the armhole chain by total arc-length --
    # armhole_cf_side is V2->hinge (stays with center-front), armhole_upper_side
    # is hinge->V3 (rotates). Works identically whether armhole is 1
    # primitive (straight panel) or many (curved panel).
    hinge, armhole_cf_side, armhole_upper_side = split_chain_by_arc_length(armhole_chain, 1 / 3)

    dart_point = fraction_along(side_seam, side_seam_t)
    hem_foot = perpendicular_foot(apex, hem)
    alpha = angle_at_apex(apex, hem_foot, hinge)
    a = (hinge - apex).norm()

    # vertex classification (Phase C step 1) -- these are already known by
    # construction, this just double-checks classify_point agrees. Also
    # checks every joint in the armhole chain's two sides (Phase D note on
    # classification) -- the hinge itself is excluded, it's deliberately on
    # the classification ray (the pivot, not a point that needs a side).
    expected = {
        "V0": (V0, Region.CENTER_FRONT),
        "V1": (V1, Region.CENTER_FRONT),
        "V1b": (V1b, Region.CENTER_FRONT),
        "V2": (V2, Region.CENTER_FRONT),
        "V3": (V3, Region.UPPER_SIDE),
        "V4": (V4, Region.LOWER_SIDE),
    }
    for name, (v, want) in expected.items():
        got = classify_point(v, apex, hinge, dart_point, hem_foot)
        assert got == want, f"{name} classified as {got}, expected {want}"
    for joint in [p.start for p in armhole_cf_side]:
        got = classify_point(joint, apex, hinge, dart_point, hem_foot)
        assert got == Region.CENTER_FRONT, f"armhole cf-side joint classified as {got}"
    for joint in [p.end for p in armhole_upper_side]:
        got = classify_point(joint, apex, hinge, dart_point, hem_foot)
        assert got == Region.UPPER_SIDE, f"armhole upper-side joint classified as {got}"

    theta = dart_angle(alpha, a, t)
    trans = lower_region_translation(apex, hem_foot, dart_point, alpha, theta, t)

    # the three motions (Phase C step 2)
    rotate_fn = lambda p: rotate_about(p, hinge, theta)  # noqa: E731
    translate_fn = lambda p: p + trans  # noqa: E731

    armhole_upper_side_rotated = transform_chain(armhole_upper_side, rotate_fn)
    rotated_V3 = armhole_upper_side_rotated[-1].end
    rotated_dart_point = rotate_about(dart_point, hinge, theta)
    translated_dart_point = dart_point + trans
    translated_V4 = V4 + trans
    translated_hem_foot = hem_foot + trans
    # V0, V1, V1b, V2, hinge, armhole_cf_side, hem_foot (center-front's
    # copy) are identity -- unchanged.

    # dart notch (Phase C step 3): apex' is the exact convergence point
    # (used internally -- see module docstring), but the drawn tip is
    # anchored to the TRUE, fixed apex (the marked bust point, which never
    # moves) pulled toward the dart opening by min(target, gap). Anchoring
    # to apex' instead would visually drift the tip away from the marked
    # apex as theta grows, since apex' is theta's own rotated copy of it.
    apex_prime = rotate_about(apex, hinge, theta)
    gap = (rotated_dart_point - translated_dart_point).norm()
    target_setback = torch.tensor(tip_setback_inches * INCH_TO_CM * panel.transform.scale)
    setback = torch.minimum(target_setback, gap)
    dart_midpoint = (rotated_dart_point + translated_dart_point) / 2
    pullback_dir = dart_midpoint - apex
    pullback_dir = pullback_dir / pullback_dir.norm()
    tip = apex + setback * pullback_dir

    # extend the new (translated) hem along its own direction until it
    # meets the center-front line -- that intersection replaces V0/hem_foot.
    new_corner = line_intersection(translated_hem_foot, translated_hem_foot - translated_V4, V0, V1 - V0)
    cf_extension = float(((new_corner - V0) * (V1 - V0)).sum() / (V1 - V0).norm())

    boundary: list[Line | Arc] = [
        Line(new_corner, V1),
        *collar_chain,
        Line(V1b, V2),
        *armhole_cf_side,
        *armhole_upper_side_rotated,
        Line(rotated_V3, rotated_dart_point),
        Line(rotated_dart_point, tip),
        Line(tip, translated_dart_point),
        Line(translated_dart_point, translated_V4),
        Line(translated_V4, translated_hem_foot),
        Line(translated_hem_foot, new_corner),
    ]

    return DartResult(
        boundary=boundary,
        apex=apex,
        hinge=hinge,
        dart_point=dart_point,
        hem_foot=hem_foot,
        apex_prime=apex_prime,
        tip=tip,
        v0_original=V0,
        rotated_V3=rotated_V3,
        rotated_dart_point=rotated_dart_point,
        translated_dart_point=translated_dart_point,
        translated_V4=translated_V4,
        translated_hem_foot=translated_hem_foot,
        new_corner=new_corner,
        alpha=alpha,
        theta=theta,
        t=t,
        trans=trans,
        cf_extension=cf_extension,
        armhole_cf_side=armhole_cf_side,
        armhole_upper_side_rotated=armhole_upper_side_rotated,
        collar_chain=collar_chain,
    )


def undo_dart(result: DartResult) -> dict[str, object]:
    """Invert insert_dart: rotate the upper-side points/chain by -theta
    about hinge, translate the lower-side points by -trans, and re-derive
    V0 by the same hem/center-front line intersection insert_dart used (not
    by just handing back the stored original -- this way undo_dart actually
    has to reconstruct it, not cheat).

    Goes through apex_prime, not the drawn tip -- tip is pulled back from
    apex_prime by a setback that isn't cleanly invertible (it depends on
    apex_prime itself), so apex_prime has to be available directly rather
    than recovered from tip.
    """
    hinge = result.hinge
    theta = result.theta
    trans = result.trans

    V3 = rotate_about(result.rotated_V3, hinge, -theta)
    dart_point_from_upper = rotate_about(result.rotated_dart_point, hinge, -theta)
    apex = rotate_about(result.apex_prime, hinge, -theta)
    dart_point_from_lower = result.translated_dart_point - trans
    V4 = result.translated_V4 - trans
    hem_foot = result.translated_hem_foot - trans

    V1 = result.collar_chain[0].start
    V1b = result.collar_chain[-1].end
    V2 = result.armhole_cf_side[0].start
    V0 = line_intersection(hem_foot, hem_foot - V4, result.new_corner, V1 - result.new_corner)

    armhole_upper_side = transform_chain(result.armhole_upper_side_rotated, lambda p: rotate_about(p, hinge, -theta))
    armhole_reconstructed = result.armhole_cf_side + armhole_upper_side

    return {
        "V0": V0,
        "V1": V1,
        "V1b": V1b,
        "V2": V2,
        "V3": V3,
        "V4": V4,
        "apex": apex,
        "dart_point_from_upper": dart_point_from_upper,
        "dart_point_from_lower": dart_point_from_lower,
        "hem_foot": hem_foot,
        "armhole_reconstructed": armhole_reconstructed,
    }
