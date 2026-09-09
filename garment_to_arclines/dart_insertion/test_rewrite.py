"""Phase F tests: the AddDartSpec/apply_add_dart wrapper (rewrite.py) must
reproduce insert_dart exactly -- this phase adds no new geometry, only a
D4D-shaped calling convention (spec in, result out) around the
already-validated transform. No candidate search: apex/side_seam_t are
fixed, known inputs, same as Phase E.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from conversion.loader import load_panel  # noqa: E402

from dart_insertion.chains import chain_length  # noqa: E402
from dart_insertion.rewrite import (  # noqa: E402
    AddDartSpec,
    RemoveDartSpec,
    apply_add_dart,
    apply_remove_dart,
    generate_add_dart_specs,
)
from dart_insertion.synthetic_panel import build_synthetic_torso_panel, edge_by_label, edges_by_label  # noqa: E402
from dart_insertion.transform import inches_to_t, insert_dart  # noqa: E402

SPEC_PATH = "../GarmentCodeData/garments_5000_0/default_body/data/rand_5DVS167AF5/rand_5DVS167AF5_specification.json"
PANEL_NAME = "right_ftorso"
EDGE_RELABEL = {0: "center_front", 1: "collar", 2: "shoulder", 3: "armhole", 4: "side_seam", 5: "hem"}

REAL_APEX_RAW = (-11.0, 6.0)
REAL_SIDE_SEAM_T = 0.65

SYNTH_APEX_RAW = (-13.0, 29.0)
SYNTH_SIDE_SEAM_T = 0.25


def _real_panel():
    panel = load_panel(SPEC_PATH, PANEL_NAME)
    panel.edge_labels = dict(EDGE_RELABEL)
    return panel, REAL_APEX_RAW, REAL_SIDE_SEAM_T


def _synthetic_panel():
    return build_synthetic_torso_panel(curved=True), SYNTH_APEX_RAW, SYNTH_SIDE_SEAM_T


CASES = [_synthetic_panel, _real_panel]
CASE_IDS = ["curved_synthetic", "real_panel"]


@pytest.mark.parametrize("build", CASES, ids=CASE_IDS)
def test_apply_add_dart_matches_insert_dart(build):
    panel, apex_raw, side_seam_t = build()
    t = inches_to_t(panel, 0.5)

    spec = AddDartSpec(apex_raw=apex_raw, side_seam_t=side_seam_t, t=t)
    via_rewrite = apply_add_dart(panel, spec)
    direct = insert_dart(panel, apex_raw, side_seam_t, t)

    assert torch.allclose(via_rewrite.apex, direct.apex)
    assert torch.allclose(via_rewrite.tip, direct.tip)
    assert torch.allclose(via_rewrite.theta, direct.theta)
    assert len(via_rewrite.boundary) == len(direct.boundary)


@pytest.mark.parametrize("build", CASES, ids=CASE_IDS)
def test_add_dart_produces_closed_boundary(build):
    panel, apex_raw, side_seam_t = build()
    t = inches_to_t(panel, 0.5)
    spec = AddDartSpec(apex_raw=apex_raw, side_seam_t=side_seam_t, t=t)
    result = apply_add_dart(panel, spec)

    boundary = result.boundary
    assert torch.allclose(boundary[-1].end, boundary[0].start, atol=1e-5)
    for prim in boundary:
        assert not bool(prim.start.isnan().any())
        assert not bool(prim.end.isnan().any())


@pytest.mark.parametrize("build", CASES, ids=CASE_IDS)
def test_add_dart_forced_fire_is_unconditional(build):
    """No accept/reject logic exists yet -- firing always produces a
    different boundary than the un-darted original (t=0 baseline)."""
    panel, apex_raw, side_seam_t = build()
    t = inches_to_t(panel, 0.5)
    spec = AddDartSpec(apex_raw=apex_raw, side_seam_t=side_seam_t, t=t)

    fired = apply_add_dart(panel, spec)
    unfired = apply_add_dart(panel, AddDartSpec(apex_raw=apex_raw, side_seam_t=side_seam_t, t=0.0))

    assert not torch.allclose(fired.tip, unfired.tip)


@pytest.mark.parametrize("build", CASES, ids=CASE_IDS)
def test_generate_add_dart_specs_proposes_one_candidate_for_eligible_panel(build):
    panel, apex_raw, side_seam_t = build()
    specs = generate_add_dart_specs(panel, apex_raw, side_seam_t, t_inches=0.5)

    assert len(specs) == 1
    assert specs[0].apex_raw == apex_raw
    assert specs[0].side_seam_t == side_seam_t
    assert specs[0].t == inches_to_t(panel, 0.5)


@pytest.mark.parametrize("build", CASES, ids=CASE_IDS)
def test_generate_add_dart_specs_proposes_nothing_for_ineligible_panel(build):
    """A panel missing even one of the required edge roles isn't
    structured for this dart -- mirrors RemoveHole proposing nothing when
    there's no qualifying loop, rather than firing on bad input."""
    panel, apex_raw, side_seam_t = build()
    panel.edge_labels = {k: v for k, v in panel.edge_labels.items() if v != "side_seam"}

    assert generate_add_dart_specs(panel, apex_raw, side_seam_t) == []


@pytest.mark.parametrize("build", CASES, ids=CASE_IDS)
def test_apply_remove_dart_reverses_apply_add_dart(build):
    """Reversibility (D4D.pdf Table 1): firing AddDart then RemoveDart
    should recover the original panel's vertices, within the same
    tolerance test_transform.py's test_reversible already established for
    undo_dart directly -- this only checks the D4D-shaped wrapper calls
    through to it correctly, not new geometry."""
    panel, apex_raw, side_seam_t = build()
    t = inches_to_t(panel, 0.5)

    fired = apply_add_dart(panel, AddDartSpec(apex_raw=apex_raw, side_seam_t=side_seam_t, t=t))
    reversed_ = apply_remove_dart(fired, RemoveDartSpec())
    rec = reversed_.reconstructed

    assert torch.allclose(rec["apex"], fired.apex, atol=1e-5)
    assert torch.allclose(rec["dart_point_from_upper"], fired.dart_point, atol=1e-5)
    assert torch.allclose(rec["dart_point_from_lower"], fired.dart_point, atol=1e-5)
    assert torch.allclose(rec["hem_foot"], fired.hem_foot, atol=1e-5)


@pytest.mark.parametrize("build", CASES, ids=CASE_IDS)
def test_apply_remove_dart_boundary_matches_original_panel(build):
    """The assembled boundary should trace the same shape as the ORIGINAL,
    never-darted panel -- checked via each of the 6 role-edges' endpoints
    and the armhole's total arc length, NOT by comparing primitive lists
    directly. The hinge split introduced while inserting the dart
    (split_chain_by_arc_length) is permanent: undo_dart reverses the rigid
    motion, not the split itself, so the reconstructed armhole legitimately
    has one more primitive than the original (M+1 vs M) -- same as any D4D
    Split that's never explicitly Merged back. Flagging this rather than
    hiding it: full topological reversibility (exact primitive-count
    restoration, matching D4D's Split/Merge symmetry) isn't achieved here,
    only geometric/positional reversibility."""
    panel, apex_raw, side_seam_t = build()
    t = inches_to_t(panel, 0.5)

    fired = apply_add_dart(panel, AddDartSpec(apex_raw=apex_raw, side_seam_t=side_seam_t, t=t))
    reversed_ = apply_remove_dart(fired, RemoveDartSpec())

    boundary = reversed_.boundary
    assert torch.allclose(boundary[-1].end, boundary[0].start, atol=1e-5)
    for prim in boundary:
        assert not bool(prim.start.isnan().any())
        assert not bool(prim.end.isnan().any())

    rec = reversed_.reconstructed
    cf_edge = edge_by_label(panel, "center_front")
    shoulder = edge_by_label(panel, "shoulder")
    armhole_chain = edges_by_label(panel, "armhole")
    hem = edge_by_label(panel, "hem")
    side_seam = edge_by_label(panel, "side_seam")

    assert torch.allclose(rec["V0"], cf_edge.start, atol=1e-5)
    assert torch.allclose(rec["V1"], cf_edge.end, atol=1e-5)
    assert torch.allclose(rec["V2"], shoulder.end, atol=1e-5)
    assert torch.allclose(rec["V4"], side_seam.end, atol=1e-5)
    assert torch.allclose(rec["V4"], hem.start, atol=1e-5)
    assert torch.allclose(rec["V0"], hem.end, atol=1e-5)

    # armhole: endpoints + total arc length match, not primitive count (see
    # docstring -- the hinge split is permanent)
    recon_armhole = rec["armhole_reconstructed"]
    assert torch.allclose(recon_armhole[0].start, armhole_chain[0].start, atol=1e-5)
    assert torch.allclose(recon_armhole[-1].end, armhole_chain[-1].end, atol=1e-5)
    assert torch.allclose(chain_length(recon_armhole), chain_length(armhole_chain), atol=1e-4)
