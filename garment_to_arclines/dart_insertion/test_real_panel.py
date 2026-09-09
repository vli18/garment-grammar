"""Phase E, Step 3: the full Phase C + Phase D assertion suite, re-run
against a REAL GarmentCodeData panel (rand_5DVS167AF5, right_ftorso --
dartless, hand-identified edges, see real_panel.py / real_panel_dart.py)
instead of the synthetic one.

Kept in its own file, deliberately not merged into test_transform.py: if
something here fails while test_transform.py stays green, that isolates
the failure to something about this real panel's geometry (vertex
ordering, real proportions, curve fitting quirks) rather than a
transform-logic bug -- those are already proven on the synthetic suite.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from d4descent.objects.arclines import Arc  # noqa: E402

from conversion.loader import load_panel  # noqa: E402
from dart_insertion.chains import chain_length, sample_chain  # noqa: E402
from dart_insertion.equations import outward_unit  # noqa: E402
from dart_insertion.geometry import polygon_area, rotate_about  # noqa: E402
from dart_insertion.regions import Region, classify_point  # noqa: E402
from dart_insertion.transform import inches_to_t, insert_dart, undo_dart  # noqa: E402

SPEC_PATH = "../GarmentCodeData/garments_5000_0/default_body/data/rand_5DVS167AF5/rand_5DVS167AF5_specification.json"
PANEL_NAME = "right_ftorso"
EDGE_RELABEL = {0: "center_front", 1: "collar", 2: "shoulder", 3: "armhole", 4: "side_seam", 5: "hem"}

APEX_RAW = (-11.0, 6.0)
SIDE_SEAM_T = 0.65
INCH_STEPS = [0.0, 0.25, 0.5, 0.75, 1.0]  # domain ceiling here is ~7.1", plenty of margin

TIGHT = 1e-5


def _panel():
    data = load_panel(SPEC_PATH, PANEL_NAME)
    data.edge_labels = dict(EDGE_RELABEL)
    return data


def _t(panel, inches):
    return inches_to_t(panel, inches)


@pytest.mark.parametrize("inches", INCH_STEPS)
def test_ray_lengths_preserved(inches):
    """Assertion 2."""
    panel = _panel()
    r = insert_dart(panel, APEX_RAW, SIDE_SEAM_T, _t(panel, inches))
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


def test_apex_fixed():
    """Assertion 3."""
    panel = _panel()
    apexes = [insert_dart(panel, APEX_RAW, SIDE_SEAM_T, _t(panel, inches)).apex for inches in INCH_STEPS]
    for a in apexes[1:]:
        assert torch.allclose(a, apexes[0], atol=TIGHT)


@pytest.mark.parametrize("inches", INCH_STEPS)
def test_armhole_length_preserved(inches):
    """Assertion 4: real 26-primitive cubic-fitted armhole chain."""
    panel = _panel()
    armhole_chain = [panel.shape.primitives[i] for i in panel.edge_map[3]]
    original_len = chain_length(armhole_chain)

    r = insert_dart(panel, APEX_RAW, SIDE_SEAM_T, _t(panel, inches))
    new_len = chain_length(r.armhole_cf_side) + chain_length(r.armhole_upper_side_rotated)
    assert torch.isclose(new_len, original_len, atol=TIGHT)


@pytest.mark.parametrize("inches", INCH_STEPS)
def test_boundary_closed(inches):
    """Assertion 7."""
    panel = _panel()
    r = insert_dart(panel, APEX_RAW, SIDE_SEAM_T, _t(panel, inches))
    shape = r.to_shape()
    n = len(shape.primitives)
    for i in range(n):
        cur, nxt = shape.primitives[i], shape.primitives[(i + 1) % n]
        assert torch.allclose(cur.end, nxt.start, atol=TIGHT), f"gap between edge {i} and {(i + 1) % n}"


@pytest.mark.parametrize("inches", INCH_STEPS)
def test_translation_components(inches):
    """Assertion 8."""
    panel = _panel()
    side_seam = panel.shape.primitives[panel.edge_map[4][0]]
    V4 = side_seam.end

    r = insert_dart(panel, APEX_RAW, SIDE_SEAM_T, _t(panel, inches))

    down_unit = r.hem_foot - r.apex
    down_unit = down_unit / down_unit.norm()
    out_unit = outward_unit(down_unit, r.dart_point - r.apex)

    disp_v4 = r.translated_V4 - V4
    disp_hem_foot = r.translated_hem_foot - r.hem_foot
    disp_dart_point = r.translated_dart_point - r.dart_point
    for disp in (disp_v4, disp_hem_foot, disp_dart_point):
        assert torch.allclose(disp, r.trans, atol=TIGHT)

    t = _t(panel, inches)
    outward_component = (r.trans * out_unit).sum()
    down_component = (r.trans * down_unit).sum()
    assert torch.isclose(outward_component, torch.tensor(t), atol=TIGHT)
    expected_down = t * torch.tan(r.alpha + r.theta / 2)
    assert torch.isclose(down_component, expected_down, atol=TIGHT)


def test_zero_state():
    """Assertion 1."""
    panel = _panel()
    orig_area = polygon_area(sample_chain(panel.shape.primitives))

    r = insert_dart(panel, APEX_RAW, SIDE_SEAM_T, 0.0)
    new_area = polygon_area(sample_chain(r.boundary))
    assert torch.isclose(new_area, orig_area, atol=TIGHT)

    cf_edge = panel.shape.primitives[panel.edge_map[0][0]]
    collar_chain = [panel.shape.primitives[i] for i in panel.edge_map[1]]
    shoulder = panel.shape.primitives[panel.edge_map[2][0]]
    armhole_chain = [panel.shape.primitives[i] for i in panel.edge_map[3]]
    side_seam = panel.shape.primitives[panel.edge_map[4][0]]
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
    assert torch.allclose(r.tip, r.apex, atol=TIGHT)
    assert torch.allclose(r.rotated_dart_point, r.dart_point, atol=TIGHT)
    assert torch.allclose(r.translated_dart_point, r.dart_point, atol=TIGHT)


@pytest.mark.parametrize("inches", INCH_STEPS)
def test_measured_dart_angle_matches_computed(inches):
    """Assertion 6."""
    panel = _panel()
    r = insert_dart(panel, APEX_RAW, SIDE_SEAM_T, _t(panel, inches))

    if inches == 0.0:
        assert torch.isclose(r.theta, torch.tensor(0.0), atol=TIGHT)
        return

    v1 = r.rotated_dart_point - r.apex_prime
    v2 = r.translated_dart_point - r.apex_prime
    cos_angle = (v1 * v2).sum() / (v1.norm() * v2.norm())
    measured = torch.acos(cos_angle.clamp(-1.0, 1.0))
    assert torch.isclose(measured, r.theta.abs(), atol=TIGHT)


@pytest.mark.parametrize("inches", INCH_STEPS)
def test_reversible(inches):
    """Assertion 5."""
    panel = _panel()
    cf_edge = panel.shape.primitives[panel.edge_map[0][0]]
    collar_chain = [panel.shape.primitives[i] for i in panel.edge_map[1]]
    shoulder = panel.shape.primitives[panel.edge_map[2][0]]
    armhole_chain = [panel.shape.primitives[i] for i in panel.edge_map[3]]
    side_seam = panel.shape.primitives[panel.edge_map[4][0]]
    V0, V1, V1b, V2, V3, V4 = (
        cf_edge.start,
        cf_edge.end,
        collar_chain[-1].end,
        shoulder.end,
        armhole_chain[-1].end,
        side_seam.end,
    )

    r = insert_dart(panel, APEX_RAW, SIDE_SEAM_T, _t(panel, inches))
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

    recon = rec["armhole_reconstructed"]
    assert torch.allclose(recon[0].start, armhole_chain[0].start, atol=TIGHT)
    assert torch.allclose(recon[-1].end, armhole_chain[-1].end, atol=TIGHT)
    assert torch.isclose(chain_length(recon), chain_length(armhole_chain), atol=TIGHT)


def _sample_one(p, tv: float) -> torch.Tensor:
    if isinstance(p, Arc):
        return p.sample(torch.tensor(tv)).reshape(2)
    return p.start + tv * (p.end - p.start)


@pytest.mark.parametrize("chain_label", ["collar", "armhole"])
@pytest.mark.parametrize("inches", [0.25, 0.5, 0.75])
def test_arc_chain_transforms_correctly(chain_label, inches):
    """Assertion 9, on the real panel's actual arc-fitted collar (1 arc,
    circle) and armhole (26 arcs, cubic) chains."""
    from dart_insertion.chains import transform_chain

    panel = _panel()
    r = insert_dart(panel, APEX_RAW, SIDE_SEAM_T, _t(panel, inches))
    hinge, theta, trans = r.hinge, r.theta, r.trans

    edge_idx = 1 if chain_label == "collar" else 3
    chain = [panel.shape.primitives[i] for i in panel.edge_map[edge_idx]]

    for motion_name, fn in [
        ("rotate", lambda p: rotate_about(p, hinge, theta)),
        ("translate", lambda p: p + trans),
    ]:
        transformed = transform_chain(chain, fn)
        for i in range(len(transformed) - 1):
            assert torch.allclose(transformed[i].end, transformed[i + 1].start, atol=TIGHT)
        for orig_p, new_p in zip(chain, transformed):
            for tv in [0.0, 0.25, 0.5, 0.75, 1.0]:
                direct = fn(_sample_one(orig_p, tv))
                indirect = _sample_one(new_p, tv)
                assert torch.allclose(direct, indirect, atol=TIGHT), f"{motion_name} {chain_label}: mismatch at t={tv}"


def test_hinge_at_third_arc_length():
    """Assertion 10, on the real 26-primitive armhole chain."""
    from dart_insertion.chains import split_chain_by_arc_length

    panel = _panel()
    armhole = [panel.shape.primitives[i] for i in panel.edge_map[3]]

    hinge, cf_side, upper_side = split_chain_by_arc_length(armhole, 1 / 3)
    total = chain_length(armhole)

    assert torch.isclose(chain_length(upper_side), total / 3, atol=TIGHT)
    assert torch.isclose(chain_length(cf_side), total * 2 / 3, atol=TIGHT)

    recon = cf_side + upper_side
    assert torch.allclose(recon[0].start, armhole[0].start, atol=TIGHT)
    assert torch.allclose(recon[-1].end, armhole[-1].end, atol=TIGHT)
    for i in range(len(recon) - 1):
        assert torch.allclose(recon[i].end, recon[i + 1].start, atol=TIGHT)

    chord_third = armhole[0].start + (armhole[-1].end - armhole[0].start) * (2 / 3)
    assert (hinge - chord_third).norm() > 0.01


def test_armhole_chain_classification_consistent():
    """Note on classification, real panel."""
    from dart_insertion.chains import split_chain_by_arc_length
    from dart_insertion.geometry import fraction_along, perpendicular_foot
    import numpy as np

    panel = _panel()
    armhole = [panel.shape.primitives[i] for i in panel.edge_map[3]]
    hem = panel.shape.primitives[panel.edge_map[5][0]]
    side_seam = panel.shape.primitives[panel.edge_map[4][0]]

    apex = torch.tensor(panel.transform.apply(np.array(APEX_RAW)), dtype=torch.float32)
    hinge, cf_side, upper_side = split_chain_by_arc_length(armhole, 1 / 3)
    dart_point = fraction_along(side_seam, SIDE_SEAM_T)
    hem_foot = perpendicular_foot(apex, hem)

    joints_cf = [p.start for p in cf_side]
    joints_upper = [p.end for p in upper_side]

    for j in joints_cf:
        assert classify_point(j, apex, hinge, dart_point, hem_foot) == Region.CENTER_FRONT
    for j in joints_upper:
        assert classify_point(j, apex, hinge, dart_point, hem_foot) == Region.UPPER_SIDE
