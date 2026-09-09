"""Phase C/D: pytest assertions for the dart-insertion transform.

Assertions 1-8 (Phase C) are parametrized over curved in [False, True] --
the panel with straight edges and the panel with real arc chains (collar,
armhole) -- confirming the same guarantees hold either way, per Phase D
Step 3. Assertions 9 and 10 are arc-chain-specific and only meaningful on
curved=True.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from d4descent.objects.arclines import Arc  # noqa: E402

from dart_insertion.chains import chain_length, sample_chain, split_chain_by_arc_length, transform_chain  # noqa: E402
from dart_insertion.equations import outward_unit  # noqa: E402
from dart_insertion.geometry import perpendicular_foot, polygon_area, rotate_about  # noqa: E402
from dart_insertion.regions import Region, classify_point  # noqa: E402
from dart_insertion.synthetic_panel import build_synthetic_torso_panel, edge_by_label, edges_by_label  # noqa: E402
from dart_insertion.transform import insert_dart, undo_dart  # noqa: E402

APEX_RAW = (-13.0, 29.0)
SIDE_SEAM_T = 0.25
T_VALUES = [0.0, 0.25, 0.5, 0.7]
# t_max = a*(1+sin(alpha)) depends on where the hinge actually lands, which
# differs between panels: curved's arc-length-based hinge sits at a
# different point than straight's chord-based one (1/3 of the curved path
# != 1/3 of the straight chord), giving a smaller a and alpha and thus a
# LOWER ceiling -- t_max ~= 0.97 straight, ~= 0.80 curved. 0.7 clears both.
CURVED_VALUES = [False, True]

# tight tolerance for exact-preservation identities (algebraic consequences
# of rotation/translation being isometries -- not measurement-derived, so
# float32 roundoff is the only source of error)
TIGHT = 1e-5


@pytest.mark.parametrize("curved", CURVED_VALUES)
@pytest.mark.parametrize("t", T_VALUES)
def test_ray_lengths_preserved(t, curved):
    """Assertion 2: both copies of each of the 3 rays equal the original
    ray's length, for every t. CF's copies are literally unmoved (identity)
    so those are trivial; the interesting half is the upper-side/lower-side
    copies, which must preserve length because rotation and translation are
    isometries."""
    panel = build_synthetic_torso_panel(curved=curved)
    r = insert_dart(panel, APEX_RAW, SIDE_SEAM_T, t)
    apex, hinge, dart_point, hem_foot, theta, trans = (
        r.apex,
        r.hinge,
        r.dart_point,
        r.hem_foot,
        r.theta,
        r.trans,
    )

    line1_len = (hem_foot - apex).norm()
    lower_copy1 = ((hem_foot + trans) - (apex + trans)).norm()
    assert torch.isclose(lower_copy1, line1_len, atol=TIGHT)

    line2_len = (hinge - apex).norm()
    upper_copy2 = (hinge - rotate_about(apex, hinge, theta)).norm()
    assert torch.isclose(upper_copy2, line2_len, atol=TIGHT)

    line3_len = (dart_point - apex).norm()
    upper_copy3 = (rotate_about(dart_point, hinge, theta) - rotate_about(apex, hinge, theta)).norm()
    lower_copy3 = ((dart_point + trans) - (apex + trans)).norm()
    assert torch.isclose(upper_copy3, line3_len, atol=TIGHT)
    assert torch.isclose(lower_copy3, line3_len, atol=TIGHT)


@pytest.mark.parametrize("curved", CURVED_VALUES)
def test_apex_fixed(curved):
    """Assertion 3: apex coordinates are unchanged for every t."""
    panel = build_synthetic_torso_panel(curved=curved)
    apexes = [insert_dart(panel, APEX_RAW, SIDE_SEAM_T, t).apex for t in T_VALUES]
    for a in apexes[1:]:
        assert torch.allclose(a, apexes[0], atol=TIGHT)


@pytest.mark.parametrize("curved", CURVED_VALUES)
@pytest.mark.parametrize("t", T_VALUES)
def test_armhole_length_preserved(t, curved):
    """Assertion 4: total arc-length of the armhole chain (CF's unrotated
    portion + upper-side's rotated portion) is unchanged before vs. after.
    On curved=True this is real arc-length across many primitives, not a
    single chord -- generalizes the straight-panel case exactly."""
    panel = build_synthetic_torso_panel(curved=curved)
    armhole_chain = edges_by_label(panel, "armhole")
    original_len = chain_length(armhole_chain)

    r = insert_dart(panel, APEX_RAW, SIDE_SEAM_T, t)
    new_len = chain_length(r.armhole_cf_side) + chain_length(r.armhole_upper_side_rotated)
    assert torch.isclose(new_len, original_len, atol=TIGHT)


@pytest.mark.parametrize("curved", CURVED_VALUES)
@pytest.mark.parametrize("t", T_VALUES)
def test_boundary_closed(t, curved):
    """Assertion 7: the output boundary is a single closed loop -- every
    edge's end exactly meets the next edge's start, arcs included."""
    panel = build_synthetic_torso_panel(curved=curved)
    r = insert_dart(panel, APEX_RAW, SIDE_SEAM_T, t)
    shape = r.to_shape()
    n = len(shape.primitives)
    for i in range(n):
        cur, nxt = shape.primitives[i], shape.primitives[(i + 1) % n]
        assert torch.allclose(cur.end, nxt.start, atol=TIGHT), f"gap between edge {i} and {(i + 1) % n}"


@pytest.mark.parametrize("curved", CURVED_VALUES)
@pytest.mark.parametrize("t", T_VALUES)
def test_translation_components(t, curved):
    """Assertion 8: the lower-side region's translation decomposes into
    exactly t in the outward direction and t*tan(alpha + theta/2) in the
    downward direction. Also checks the side-seam point, hem_foot, and V4
    all receive the identical displacement (one rigid translation)."""
    panel = build_synthetic_torso_panel(curved=curved)
    side_seam = edge_by_label(panel, "side_seam")
    V4 = side_seam.end

    r = insert_dart(panel, APEX_RAW, SIDE_SEAM_T, t)

    down_unit = r.hem_foot - r.apex
    down_unit = down_unit / down_unit.norm()
    out_unit = outward_unit(down_unit, r.dart_point - r.apex)

    disp_v4 = r.translated_V4 - V4
    disp_hem_foot = r.translated_hem_foot - r.hem_foot
    disp_dart_point = r.translated_dart_point - r.dart_point
    for disp in (disp_v4, disp_hem_foot, disp_dart_point):
        assert torch.allclose(disp, r.trans, atol=TIGHT)

    outward_component = (r.trans * out_unit).sum()
    down_component = (r.trans * down_unit).sum()

    assert torch.isclose(outward_component, torch.tensor(t), atol=TIGHT)
    expected_down = t * torch.tan(r.alpha + r.theta / 2)
    assert torch.isclose(down_component, expected_down, atol=TIGHT)


@pytest.mark.parametrize("curved", CURVED_VALUES)
def test_zero_state(curved):
    """Assertion 1: t=0 returns a panel geometrically identical to the
    input. Measured as enclosed polygon area (shoelace over sampled
    boundary points, arcs included via chains.sample_chain) -- the standard
    notion of "same 2D shape" for a filled fabric panel -- rather than raw
    vertex-list equality, because the dart notch's tip coincides with apex
    at t=0 and both dart-point copies coincide too, so the notch is a
    genuine zero-area retrace, not a different shape. Also directly checks
    every "real" (non-notch) vertex matches vertex-for-vertex.
    """
    panel = build_synthetic_torso_panel(curved=curved)
    orig_area = polygon_area(sample_chain(panel.shape.primitives))

    r = insert_dart(panel, APEX_RAW, SIDE_SEAM_T, 0.0)
    new_area = polygon_area(sample_chain(r.boundary))
    assert torch.isclose(new_area, orig_area, atol=TIGHT)

    cf_edge = edge_by_label(panel, "center_front")
    collar_chain = edges_by_label(panel, "collar")
    shoulder = edge_by_label(panel, "shoulder")
    armhole_chain = edges_by_label(panel, "armhole")
    side_seam = edge_by_label(panel, "side_seam")
    V0, V1, V1b, V2, V3, V4 = (
        cf_edge.start,
        cf_edge.end,
        collar_chain[-1].end,
        shoulder.end,
        armhole_chain[-1].end,
        side_seam.end,
    )
    assert torch.allclose(r.new_corner, V0, atol=TIGHT)
    assert torch.allclose(r.collar_chain[0].start, V1, atol=TIGHT)
    assert torch.allclose(r.collar_chain[-1].end, V1b, atol=TIGHT)
    assert torch.allclose(r.armhole_cf_side[0].start, V2, atol=TIGHT)
    assert torch.allclose(r.rotated_V3, V3, atol=TIGHT)
    assert torch.allclose(r.translated_V4, V4, atol=TIGHT)
    # the notch: tip coincides with apex, and both dart-point copies coincide
    # with the original dart_point -- zero-area, as expected, not identical
    # vertex-for-vertex to "no notch at all" but geometrically inert.
    assert torch.allclose(r.tip, r.apex, atol=TIGHT)
    assert torch.allclose(r.rotated_dart_point, r.dart_point, atol=TIGHT)
    assert torch.allclose(r.translated_dart_point, r.dart_point, atol=TIGHT)


@pytest.mark.parametrize("curved", CURVED_VALUES)
@pytest.mark.parametrize("t", T_VALUES)
def test_measured_dart_angle_matches_computed(t, curved):
    """Assertion 6: the dart wedge angle measured from the output geometry
    equals theta = arcsin(sin(alpha) - t/a) - alpha.

    Measured at apex_prime (= rotate_about(apex, hinge, theta), where the
    two dart legs would meet exactly if extended all the way in -- NOT at
    the drawn `tip`, which is apex pulled back for practical drafting and
    doesn't satisfy this identity once the pullback is nonzero): the vector
    from apex_prime to the upper leg's endpoint is R(theta) applied to
    (dart_point - apex); the vector from apex_prime to the lower leg's
    endpoint is (dart_point - apex) unrotated (translation doesn't change
    direction). The angle between "a vector" and "that same vector rotated
    by theta" is theta exactly -- this isn't tautological the way measuring
    at the hinge would be (that only confirms rotate_about preserves
    distance-from-pivot); this confirms the two legs, which come from two
    DIFFERENT regions' motions, actually open by theta.
    """
    panel = build_synthetic_torso_panel(curved=curved)
    r = insert_dart(panel, APEX_RAW, SIDE_SEAM_T, t)

    if t == 0.0:
        # both legs collapse onto apex_prime -- angle is undefined (0/0),
        # and theta is exactly 0 anyway, so nothing to measure.
        assert torch.isclose(r.theta, torch.tensor(0.0), atol=TIGHT)
        return

    v1 = r.rotated_dart_point - r.apex_prime
    v2 = r.translated_dart_point - r.apex_prime
    cos_angle = (v1 * v2).sum() / (v1.norm() * v2.norm())
    measured = torch.acos(cos_angle.clamp(-1.0, 1.0))
    assert torch.isclose(measured, r.theta.abs(), atol=TIGHT)


@pytest.mark.parametrize("curved", CURVED_VALUES)
@pytest.mark.parametrize("t", T_VALUES)
def test_reversible(t, curved):
    """Assertion 5: applying the dart with t, then undo_dart, returns a
    panel geometrically identical to the original (V0-V4, apex, and on
    curved panels the armhole chain itself, not just its endpoints)."""
    panel = build_synthetic_torso_panel(curved=curved)
    cf_edge = edge_by_label(panel, "center_front")
    collar_chain = edges_by_label(panel, "collar")
    shoulder = edge_by_label(panel, "shoulder")
    armhole_chain = edges_by_label(panel, "armhole")
    side_seam = edge_by_label(panel, "side_seam")
    V0, V1, V1b, V2, V3, V4 = (
        cf_edge.start,
        cf_edge.end,
        collar_chain[-1].end,
        shoulder.end,
        armhole_chain[-1].end,
        side_seam.end,
    )

    r = insert_dart(panel, APEX_RAW, SIDE_SEAM_T, t)
    rec = undo_dart(r)

    assert torch.allclose(rec["V0"], V0, atol=TIGHT)
    assert torch.allclose(rec["V1"], V1, atol=TIGHT)
    assert torch.allclose(rec["V1b"], V1b, atol=TIGHT)
    assert torch.allclose(rec["V2"], V2, atol=TIGHT)
    assert torch.allclose(rec["V3"], V3, atol=TIGHT)
    assert torch.allclose(rec["V4"], V4, atol=TIGHT)
    assert torch.allclose(rec["apex"], r.apex, atol=TIGHT)
    assert torch.allclose(rec["dart_point_from_upper"], r.dart_point, atol=TIGHT)
    assert torch.allclose(rec["dart_point_from_lower"], r.dart_point, atol=TIGHT)
    assert torch.allclose(rec["hem_foot"], r.hem_foot, atol=TIGHT)

    # the armhole chain itself is fully recovered too, not just its endpoints
    recon = rec["armhole_reconstructed"]
    assert torch.allclose(recon[0].start, armhole_chain[0].start, atol=TIGHT)
    assert torch.allclose(recon[-1].end, armhole_chain[-1].end, atol=TIGHT)
    assert torch.isclose(chain_length(recon), chain_length(armhole_chain), atol=TIGHT)


def _sample_one(p, tv: float) -> torch.Tensor:
    if isinstance(p, Arc):
        return p.sample(torch.tensor(tv)).reshape(2)
    return p.start + tv * (p.end - p.start)


@pytest.mark.parametrize("chain_label", ["collar", "armhole"])
@pytest.mark.parametrize("t", [0.25, 0.5, 0.75])
def test_arc_chain_transforms_correctly(chain_label, t):
    """Assertion 9: rotating or translating a multi-primitive arc/line
    chain produces primitives that are the exact rotated/translated image
    of the original. Verified by sampling points on the ORIGINAL chain and
    transforming those points directly ("direct"), vs. building the new
    (same k, transformed endpoints) primitives and sampling THEM at the
    same parameters ("indirect") -- if k or the endpoints were handled
    wrong, these would diverge even when start/end still matched. Also
    checks the transformed chain stays connected.

    hinge/theta/trans come from insert_dart on the straight (curved=False)
    panel here just to get realistic, geometry-consistent motion values
    (collar and armhole have the same start/end vertices either way); the
    actual curved-panel insert_dart pipeline is exercised end-to-end by the
    tests above.
    """
    straight_panel = build_synthetic_torso_panel(curved=False)
    r = insert_dart(straight_panel, APEX_RAW, SIDE_SEAM_T, t)
    hinge, theta, trans = r.hinge, r.theta, r.trans

    curved_panel = build_synthetic_torso_panel(curved=True)
    chain = edges_by_label(curved_panel, chain_label)

    for motion_name, fn in [
        ("rotate", lambda p: rotate_about(p, hinge, theta)),
        ("translate", lambda p: p + trans),
    ]:
        transformed = transform_chain(chain, fn)

        for i in range(len(transformed) - 1):
            assert torch.allclose(transformed[i].end, transformed[i + 1].start, atol=TIGHT), (
                f"{motion_name} {chain_label}: chain disconnected after transform at joint {i}"
            )

        for orig_p, new_p in zip(chain, transformed):
            for tv in [0.0, 0.25, 0.5, 0.75, 1.0]:
                direct = fn(_sample_one(orig_p, tv))
                indirect = _sample_one(new_p, tv)
                assert torch.allclose(direct, indirect, atol=TIGHT), f"{motion_name} {chain_label}: mismatch at t={tv}"


def test_hinge_at_third_arc_length():
    """Assertion 10: the armhole hinge sits at exactly 1/3 of the TOTAL
    arc-length along the multi-primitive armhole chain -- not 1/3 of the
    straight chord, and not 1/3 by primitive count."""
    panel = build_synthetic_torso_panel(curved=True)
    armhole = edges_by_label(panel, "armhole")

    hinge, cf_side, upper_side = split_chain_by_arc_length(armhole, 1 / 3)
    total = chain_length(armhole)

    # armhole is built V2 (high) -> V3 (low); hinge is 1/3 up FROM V3, i.e.
    # 1/3 of the chain by length belongs to upper_side (hinge->V3) and 2/3
    # to cf_side (V2->hinge).
    assert torch.isclose(chain_length(upper_side), total / 3, atol=TIGHT)
    assert torch.isclose(chain_length(cf_side), total * 2 / 3, atol=TIGHT)

    # reconstruction: cf_side + upper_side retraces the original chain
    # exactly, with zero gap at every joint including the new split point.
    recon = cf_side + upper_side
    assert torch.allclose(recon[0].start, armhole[0].start, atol=TIGHT)
    assert torch.allclose(recon[-1].end, armhole[-1].end, atol=TIGHT)
    for i in range(len(recon) - 1):
        assert torch.allclose(recon[i].end, recon[i + 1].start, atol=TIGHT)

    # not the same as the naive (wrong) chord-based 1/3 point -- proves
    # this is actually using arc-length along the curve, not a straight-
    # line shortcut between the chain's endpoints.
    chord_third = armhole[0].start + (armhole[-1].end - armhole[0].start) * (2 / 3)
    assert (hinge - chord_third).norm() > 0.01


def test_armhole_chain_classification_consistent():
    """Note on classification: the hinge is the only transition point in
    the whole armhole chain -- every joint on the V2 side of the split
    classifies CENTER_FRONT, every joint on the V3 side classifies
    UPPER_SIDE. The hinge point itself is excluded (it's deliberately ON
    the classification ray, which classify_point rejects as ambiguous by
    design -- it's the pivot, not a point that needs a side)."""
    panel = build_synthetic_torso_panel(curved=True)
    armhole = edges_by_label(panel, "armhole")
    hem = edge_by_label(panel, "hem")
    side_seam = edge_by_label(panel, "side_seam")

    apex = torch.tensor(panel.transform.apply(np.array(APEX_RAW)), dtype=torch.float32)
    hinge, cf_side, upper_side = split_chain_by_arc_length(armhole, 1 / 3)
    dart_point = side_seam.start + SIDE_SEAM_T * (side_seam.end - side_seam.start)
    hem_foot = perpendicular_foot(apex, hem)

    joints_cf = [p.start for p in cf_side]  # V2 ... up to (not including) hinge
    joints_upper = [p.end for p in upper_side]  # (not including) hinge ... V3

    for j in joints_cf:
        assert classify_point(j, apex, hinge, dart_point, hem_foot) == Region.CENTER_FRONT
    for j in joints_upper:
        assert classify_point(j, apex, hinge, dart_point, hem_foot) == Region.UPPER_SIDE
