"""Synthetic front-torso-like panel for dart-insertion transform testing.

Built the same way loader.load_panel builds a PanelData from real
GarmentCodeData (shared vertex tensors at joints, PanelTransform
normalization to near-unit scale) so the dart-insertion transform sees the
same structure either way -- just straight-line edges only, no curvature,
and no GarmentCodeData subdivision to worry about.

Canonical orientation: center front vertical at x=0, hem horizontal at
y=0, increasing y toward the shoulder.
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from conversion.arc_fit import fit_curve_to_arcs  # noqa: E402
from conversion.bezier import sample_quadratic  # noqa: E402
from conversion.loader import DEFAULT_FIT_TOL_CM, PanelData, PanelTransform  # noqa: E402

from d4descent.objects.arclines import Arc, Line, Shape  # noqa: E402

# Real GarmentCodeData panels curve via Bezier control points fit into
# Arc/Line chains (see loader.py) -- these mimic that, mid-edge control
# offsets as (along-edge fraction, perpendicular fraction) exactly like
# GarmentCode's own convention (conversion/bezier.py::rel_to_abs_2d).
_COLLAR_CONTROL_REL = np.array([0.5, 0.20])
_ARMHOLE_CONTROL_REL = np.array([0.5, 0.26])

# Hexagon: center-front, collar (neckline), shoulder, armhole, side-seam,
# hem. Shoulder is its own straight edge between the top of the collar
# curve and the top of the armhole curve -- not blended into either, since
# a real shoulder seam is straight. Side seam is vertical (perpendicular to
# the horizontal hem) and shorter than center front, shortened from the
# top; armhole is short relative to both, as on a real torso panel.
# Vertices are in real cm, scaled to realistic bodice-front dimensions
# (center front ~42cm, side seam ~35cm, armhole ~21cm) so a real-world
# measurement like a 1" dart setback reads as a small, sane fraction of the
# panel rather than a third of it.
_RAW_VERTICES = 1.75 * np.array(
    [
        [0.0, 0.0],  # V0: hem / center-front corner
        [0.0, 24.0],  # V1: center-front / collar corner
        [-4.5, 30.0],  # V1b: top of collar / start of shoulder (level with V2 -- flat shoulder)
        [-10.5, 30.0],  # V2: end of shoulder / top of armhole
        [-15.0, 18.75],  # V3: armhole / side-seam corner
        [-15.0, 0.0],  # V4: side-seam / hem corner
    ]
)

_EDGE_LABELS = {
    0: "center_front",
    1: "collar",
    2: "shoulder",
    3: "armhole",
    4: "side_seam",
    5: "hem",
}


def build_synthetic_torso_panel(target_half_extent: float = 1.0, curved: bool = False) -> PanelData:
    """curved=False (default): all 6 edges are single Lines, as in Phase A-C.
    curved=True: collar and armhole become multi-primitive Arc/Line chains
    fit the same way loader.py fits real GarmentCodeData Bezier edges --
    same vertices, same panel shape, only those two edges' geometry between
    their fixed endpoints changes. shoulder stays straight either way."""
    center = (_RAW_VERTICES.min(axis=0) + _RAW_VERTICES.max(axis=0)) / 2
    half_extent = float((_RAW_VERTICES.max(axis=0) - _RAW_VERTICES.min(axis=0)).max() / 2)
    transform = PanelTransform(center=center, scale=target_half_extent / half_extent)

    normalized = transform.apply(_RAW_VERTICES)
    vertex_tensors = [torch.tensor(v, dtype=torch.float32) for v in normalized]
    n = len(vertex_tensors)

    fit_tol = DEFAULT_FIT_TOL_CM * transform.scale

    primitives: list[Line | Arc] = []
    edge_map: dict[int, list[int]] = {}
    for i in range(n):
        start_t, end_t = vertex_tensors[i], vertex_tensors[(i + 1) % n]
        control_rel = (
            _COLLAR_CONTROL_REL if i == 1 else _ARMHOLE_CONTROL_REL if i == 3 else None
        )
        if curved and control_rel is not None:
            start_arr, end_arr = normalized[i], normalized[(i + 1) % n]
            new_prims = fit_curve_to_arcs(
                lambda t, s=start_arr, e=end_arr, c=control_rel: sample_quadratic(s, e, c, t),
                start_t,
                end_t,
                fit_tol,
            )
        else:
            new_prims = [Line(start_t, end_t)]
        edge_map[i] = list(range(len(primitives), len(primitives) + len(new_prims)))
        primitives.extend(new_prims)

    return PanelData(shape=Shape(primitives), transform=transform, edge_map=edge_map, edge_labels=dict(_EDGE_LABELS))


def edge_by_label(panel: PanelData, label: str) -> Line:
    """The single Line primitive for a labeled edge (assumes no subdivision --
    only valid for edges that are still a single primitive, i.e. everything
    except collar/armhole on a curved=True panel; use edges_by_label for
    those)."""
    prims = edges_by_label(panel, label)
    assert len(prims) == 1 and isinstance(prims[0], Line), f"{label!r} is a multi-primitive chain, use edges_by_label"
    return prims[0]


def edges_by_label(panel: PanelData, label: str) -> list[Line | Arc]:
    """The full ordered chain of primitives for a labeled edge (length 1 for
    a plain Line edge, length >1 for a fitted arc chain)."""
    for edge_idx, lbl in panel.edge_labels.items():
        if lbl == label:
            return [panel.shape.primitives[i] for i in panel.edge_map[edge_idx]]
    raise KeyError(f"No edge labeled {label!r}")
