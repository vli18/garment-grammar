"""Load GarmentCodeData panels into D4D Arc-Line Shapes.

Supported: straight edges and `circle`-type edges (both exact, no
approximation), plus `quadratic`/`cubic` Bezier edges (approximated by an
adaptive Line/Arc chain, within fit_tol_cm).
"""

import json
from dataclasses import dataclass, field

import numpy as np
import torch

from d4descent.objects.arclines import Arc, Line, Shape

from .arc_fit import fit_curve_to_arcs
from .bezier import sample_cubic, sample_quadratic
from .circle_arc import circle_edge_to_k

_SUPPORTED_CURVATURE_TYPES = {"circle", "quadratic", "cubic"}
# Default max deviation between a fitted arc/line chain and the true curve,
# in cm. Justified by GarmentCodeData's own ~1cm simulation mesh resolution
# and ~3mm stitch-length-matching tolerance (both far looser than this).
DEFAULT_FIT_TOL_CM = 0.01


@dataclass
class PanelTransform:
    """Invertible normalization from original panel coordinates (cm) to
    D4D's expected near-unit scale.

    normalized = (original - center) * scale
    original   = normalized / scale + center
    """

    center: np.ndarray  # (2,), in original cm coordinates
    scale: float  # unitless

    def apply(self, points: np.ndarray) -> np.ndarray:
        return (points - self.center) * self.scale

    def invert(self, points: np.ndarray) -> np.ndarray:
        return points / self.scale + self.center


@dataclass
class PanelData:
    shape: Shape
    transform: PanelTransform
    # original edge index -> indices into shape.primitives it became
    edge_map: dict[int, list[int]] = field(default_factory=dict)
    # original edge index -> label, only present for labelled edges
    edge_labels: dict[int, str] = field(default_factory=dict)


def _fit_scale(vertices: np.ndarray, target_half_extent: float) -> float:
    """Uniform scale so the panel's bounding box half-extent (longer axis)
    equals target_half_extent."""
    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    half_extent = float((bbox_max - bbox_min).max() / 2)
    if half_extent <= 0:
        raise ValueError("Degenerate panel bounding box")
    return target_half_extent / half_extent


def load_panel(
    spec_path: str,
    panel_name: str,
    target_half_extent: float = 1.0,
    fit_tol_cm: float = DEFAULT_FIT_TOL_CM,
) -> PanelData:
    """Load a single panel from a GarmentCodeData specification.json.

    fit_tol_cm: max allowed deviation (in cm) between a fitted arc/line chain
        and the true curve, for quadratic/cubic edges (approximated -- all
        other edge types are exact).

    Raises NotImplementedError if the panel has a cubic Bezier edge (not yet
    supported).
    """
    with open(spec_path) as f:
        spec = json.load(f)

    panel = spec["pattern"]["panels"][panel_name]
    vertices = np.array(panel["vertices"], dtype=np.float64)
    edges = panel["edges"]

    for i, edge in enumerate(edges):
        curvature = edge.get("curvature")
        if curvature is not None and curvature["type"] not in _SUPPORTED_CURVATURE_TYPES:
            raise NotImplementedError(
                f"Panel '{panel_name}' edge {i} curvature type "
                f"'{curvature['type']}' is not supported yet."
            )

    center = (vertices.min(axis=0) + vertices.max(axis=0)) / 2
    scale = _fit_scale(vertices, target_half_extent)
    transform = PanelTransform(center=center, scale=scale)

    normalized_vertices = transform.apply(vertices)
    # One tensor per vertex, shared across whichever edges reference it, so
    # consecutive primitives share object identity at their joint -- D4D's
    # connectivity logic (find_loops, merge_line/merge_arc, do_simplify, ...)
    # keys off id(), not value equality.
    vertex_tensors = [torch.tensor(v, dtype=torch.float32) for v in normalized_vertices]

    fit_tol_normalized = fit_tol_cm * scale

    primitives: list[Line | Arc] = []
    edge_map: dict[int, list[int]] = {}
    edge_labels: dict[int, str] = {}
    for i, edge in enumerate(edges):
        start_idx, end_idx = edge["endpoints"]
        start_arr = normalized_vertices[start_idx]
        end_arr = normalized_vertices[end_idx]
        start = vertex_tensors[start_idx]
        end = vertex_tensors[end_idx]

        curvature = edge.get("curvature")
        if curvature is None:
            new_prims: list[Line | Arc] = [Line(start, end)]
        elif curvature["type"] == "circle":
            radius, large_arc, right = curvature["params"]
            # radius is in original cm units; scale it to match the
            # already-normalized start/end so h/k come out in normalized units.
            k = circle_edge_to_k(start_arr, end_arr, radius * scale, large_arc, right)
            new_prims = [Arc(start, end, torch.tensor(k, dtype=torch.float32))]
        elif curvature["type"] == "quadratic":
            control_rel = np.array(curvature["params"][0], dtype=np.float64)
            # relative coords are scale/translation-invariant, so sampling
            # directly with the already-normalized start/end is correct --
            # only the fit tolerance needs converting to normalized units.
            new_prims = fit_curve_to_arcs(
                lambda t: sample_quadratic(start_arr, end_arr, control_rel, t),
                start,
                end,
                fit_tol_normalized,
            )
        elif curvature["type"] == "cubic":
            c1_rel = np.array(curvature["params"][0], dtype=np.float64)
            c2_rel = np.array(curvature["params"][1], dtype=np.float64)
            new_prims = fit_curve_to_arcs(
                lambda t: sample_cubic(start_arr, end_arr, c1_rel, c2_rel, t),
                start,
                end,
                fit_tol_normalized,
            )
        else:
            raise NotImplementedError(f"Unhandled curvature type '{curvature['type']}'")

        primitives.extend(new_prims)
        edge_map[i] = list(range(len(primitives) - len(new_prims), len(primitives)))
        if "label" in edge:
            edge_labels[i] = edge["label"]

    shape = Shape(primitives)
    return PanelData(shape=shape, transform=transform, edge_map=edge_map, edge_labels=edge_labels)
