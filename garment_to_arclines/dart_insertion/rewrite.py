"""Phase F: wrap insert_dart as a D4D-style rewrite -- a (spec, apply) pair
matching D4D's ShapeRewriteSpec / apply_rewrite convention, so it's ready to
plug into SRD's accept/reject loop later.

Dart position (apex + side-seam entry point) is NOT searched here -- it's a
fixed, known input per panel, same as every prior phase (the transform's
math is specific to a side-seam dart; other entry edges are an unvalidated,
separate question for later, possibly as other rewrite *types* to choose
between). What a real optimizer will eventually decide is (1) whether to
fire this rewrite at all -- SRD's existing generic accept-if-better logic,
not built here -- and (2) how much spread (t) to use. Right now both of
those are forced: t is fixed and firing is unconditional, so this phase only
confirms the wrapper reproduces insert_dart correctly.
"""

from dataclasses import dataclass

from conversion.loader import PanelData
from d4descent.objects.arclines import Arc, Line

from .transform import DartResult, inches_to_t, insert_dart, undo_dart

REQUIRED_EDGE_LABELS = {"center_front", "collar", "shoulder", "armhole", "side_seam", "hem"}


@dataclass
class AddDartSpec:
    """Mirrors D4D's ShapeRewriteSpec shape: a small, explicit description
    of one dart-insertion call. apex_raw/side_seam_t are fixed per panel
    (not searched); t is the one currently-forced continuous parameter."""

    apex_raw: tuple[float, float]
    side_seam_t: float
    t: float


def apply_add_dart(panel: PanelData, spec: AddDartSpec) -> DartResult:
    """The 'apply_rewrite' side of the pair -- same call shape D4D uses
    (spec in, new geometry out), wrapping insert_dart directly."""
    return insert_dart(panel, spec.apex_raw, spec.side_seam_t, spec.t)


def generate_add_dart_specs(
    panel: PanelData, apex_raw: tuple[float, float], side_seam_t: float, t_inches: float = 0.5
) -> list[AddDartSpec]:
    """The 'generate_rewrite_specs' side of the pair -- but not a search.
    This transform's math depends on a full set of labeled edge roles
    (center_front/collar/shoulder/armhole/side_seam/hem) all present on the
    SAME panel at once, so there's at most one valid candidate for a given
    panel, not several to choose between -- apex_raw/side_seam_t are still
    supplied by the caller (known/derived values), not sampled here.

    Mirrors RemoveHole's pattern in D4D: propose nothing if the shape
    doesn't qualify, rather than firing on a panel that isn't structured
    for this dart."""
    if not REQUIRED_EDGE_LABELS.issubset(panel.edge_labels.values()):
        return []
    return [AddDartSpec(apex_raw=apex_raw, side_seam_t=side_seam_t, t=inches_to_t(panel, t_inches))]


@dataclass
class RemoveDartSpec:
    """The reverse of AddDartSpec, per D4D's Reversibility guideline (Table
    1: if a rule A->B exists, B->A should too). No fields: unlike AddDart,
    there's nothing left to choose -- undoing is fully determined by the
    DartResult it's given.

    KNOWN LIMITATION (see d4d_integration_notes.md): this only reverses a
    dart *just fired in the same call*, identified by its DartResult (which
    carries hinge/theta/trans -- state that isn't recoverable from a plain
    Shape's primitive list alone). It cannot reverse an arbitrary
    already-darted shape encountered later, e.g. after other rewrites ran
    in between -- that would need this state stashed somewhere recoverable
    (D4D's ShapePayload?) or re-derived from the boundary geometry, neither
    of which is built here."""


@dataclass
class UndoDartResult:
    boundary: list[Line | Arc]  # full assembled panel boundary, closed loop -- matches DartResult.boundary's contract
    reconstructed: dict[str, object]  # undo_dart's raw point-level output, kept for point-level checks


def apply_remove_dart(result: DartResult, spec: RemoveDartSpec) -> UndoDartResult:
    """The 'apply_rewrite' side of the reverse pair -- wraps undo_dart, then
    assembles its reconstructed points back into a real closed boundary
    (same 6-edge-role structure as the original un-darted panel: cf,
    collar, shoulder, armhole, side_seam, hem), so the output is an actual
    shape to draw/compare, not just a bag of points."""
    rec = undo_dart(result)
    boundary: list[Line | Arc] = [
        Line(rec["V0"], rec["V1"]),
        *result.collar_chain,  # unchanged by insert_dart, so unchanged by its inverse too
        Line(rec["V1b"], rec["V2"]),
        *rec["armhole_reconstructed"],
        Line(rec["V3"], rec["V4"]),
        Line(rec["V4"], rec["V0"]),
    ]
    return UndoDartResult(boundary=boundary, reconstructed=rec)
