"""Side-by-side visual comparison: original GarmentCodeData render vs. our
loaded D4D Shapes.

The stored Shape/PanelTransform keep our per-panel normalization (needed
for the data pipeline). For display only, panel coordinates are inverted
back to true cm and shown on a shared scale across panels, so relative
panel sizes are visually comparable -- matching how the reference PNG
shows them.
"""

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import torch

from conversion.arc_fit import fit_curve_to_arcs
from conversion.bezier import sample_cubic, sample_quadratic
from conversion.circle_arc import circle_edge_to_k
from conversion.loader import DEFAULT_FIT_TOL_CM, PanelData, load_panel
from d4descent.objects.arclines import Arc, Line, Shape
from d4descent.visualizer import LineStyle, MPLVisualizerAxes, PointStyle

from verification.ground_truth import _sample_edge as gt_sample_edge
from verification.ground_truth import sample_panel_boundary


def _true_scale_shape(data: PanelData) -> Shape:
    """Rebuild a Shape in original cm coordinates by inverting the panel's
    normalization transform. For visualization only -- not used downstream."""
    prims: list[Line | Arc] = []
    for prim in data.shape.primitives:
        start = torch.tensor(data.transform.invert(prim.start.numpy()), dtype=torch.float32)
        end = torch.tensor(data.transform.invert(prim.end.numpy()), dtype=torch.float32)
        if isinstance(prim, Arc):
            k = prim.k / data.transform.scale
            prims.append(Arc(start, end, k))
        else:
            prims.append(Line(start, end))
    return Shape(prims)


def render_comparison(
    spec_path: str,
    panel_names: list[str],
    reference_png_path: str,
    out_path: str,
    grid_shape: tuple[int, int] | None = None,
) -> None:
    n = len(panel_names)
    if grid_shape is None:
        ncols = int(np.ceil(np.sqrt(n)))
        nrows = int(np.ceil(n / ncols))
    else:
        nrows, ncols = grid_shape

    true_shapes = []
    all_points = []
    for panel_name in panel_names:
        data = load_panel(spec_path, panel_name)
        shape = _true_scale_shape(data)
        true_shapes.append((panel_name, shape))
        for prim in shape.primitives:
            all_points.append(prim.start.numpy())
            all_points.append(prim.end.numpy())
    all_points = np.array(all_points)
    margin = 0.08 * max(np.ptp(all_points[:, 0]), np.ptp(all_points[:, 1]))
    xlim = (all_points[:, 0].min() - margin, all_points[:, 0].max() + margin)
    ylim = (all_points[:, 1].min() - margin, all_points[:, 1].max() + margin)

    fig = plt.figure(figsize=(14, 7), layout="constrained")
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1])

    ax_ref = fig.add_subplot(gs[0, 0])
    ax_ref.imshow(mpimg.imread(reference_png_path))
    ax_ref.axis("off")
    ax_ref.set_title("GarmentCodeData reference")

    gs_ours = gs[0, 1].subgridspec(nrows, ncols)
    for idx, (panel_name, shape) in enumerate(true_shapes):
        r, c = divmod(idx, ncols)
        ax = fig.add_subplot(gs_ours[r, c])
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(panel_name, fontsize=9)
        MPLVisualizerAxes(ax).visualize_shape(shape)
    fig.suptitle("Our loaded D4D Shapes (true cm scale, shared axes)")

    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def render_overlay(
    spec_path: str,
    panel_names: list[str],
    out_path: str,
    grid_shape: tuple[int, int] | None = None,
) -> None:
    """Per panel: true ground-truth boundary from specification.json (filled,
    curvature honored via svgpathtools -- independent of our own conversion
    math) with our loaded D4D Shape drawn on top (blue outline). If our Shape
    is correct, the outline exactly traces the fill's edges, curves included."""
    from matplotlib.patches import Polygon

    n = len(panel_names)
    if grid_shape is None:
        ncols = int(np.ceil(np.sqrt(n)))
        nrows = int(np.ceil(n / ncols))
    else:
        nrows, ncols = grid_shape

    fig, axs = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4.5 * nrows), layout="constrained")
    axs = np.atleast_2d(axs).flatten()

    for idx, panel_name in enumerate(panel_names):
        ax = axs[idx]
        boundary = sample_panel_boundary(spec_path, panel_name)

        data = load_panel(spec_path, panel_name)
        shape = _true_scale_shape(data)

        margin = 0.12 * max(np.ptp(boundary[:, 0]), np.ptp(boundary[:, 1]))
        ax.set_xlim(boundary[:, 0].min() - margin, boundary[:, 0].max() + margin)
        ax.set_ylim(boundary[:, 1].min() - margin, boundary[:, 1].max() + margin)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(panel_name, fontsize=9)

        ax.add_patch(Polygon(boundary, closed=True, facecolor=(0.89, 0.69, 0.73), edgecolor="none", zorder=0))
        MPLVisualizerAxes(ax).visualize_shape(
            shape,
            line_style=LineStyle(color="blue", linewidth=1.2, arrowwidth=None),
            point_style=PointStyle(color="blue", radius=10),
        )

    for idx in range(n, len(axs)):
        axs[idx].axis("off")

    fig.suptitle("Ground truth (fill) vs. our D4D Shape (blue outline)")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def _fit_single_edge(start: np.ndarray, end: np.ndarray, curvature: dict | None, fit_tol_cm: float) -> list[Line | Arc]:
    """Same per-edge conversion logic as loader.load_panel, but standalone --
    for checking individual edges on panels that can't fully load yet because
    they also contain an unsupported edge type elsewhere."""
    start_t = torch.tensor(start, dtype=torch.float32)
    end_t = torch.tensor(end, dtype=torch.float32)
    if curvature is None:
        return [Line(start_t, end_t)]
    if curvature["type"] == "circle":
        radius, large_arc, right = curvature["params"]
        k = circle_edge_to_k(start, end, radius, large_arc, right)
        return [Arc(start_t, end_t, torch.tensor(k, dtype=torch.float32))]
    if curvature["type"] == "quadratic":
        control_rel = np.array(curvature["params"][0], dtype=np.float64)
        return fit_curve_to_arcs(
            lambda t: sample_quadratic(start, end, control_rel, t), start_t, end_t, fit_tol_cm
        )
    if curvature["type"] == "cubic":
        c1_rel = np.array(curvature["params"][0], dtype=np.float64)
        c2_rel = np.array(curvature["params"][1], dtype=np.float64)
        return fit_curve_to_arcs(
            lambda t: sample_cubic(start, end, c1_rel, c2_rel, t), start_t, end_t, fit_tol_cm
        )
    raise NotImplementedError(f"Unhandled curvature type '{curvature['type']}'")


def render_edge_overlay(
    spec_path: str,
    panel_name: str,
    edge_indices: list[int],
    out_path: str,
    fit_tol_cm: float = DEFAULT_FIT_TOL_CM,
) -> None:
    """Per specified edge (not the whole panel -- useful while a panel has a
    mix of supported and not-yet-supported edge types): true curve (fill)
    vs. our fitted Line/Arc chain (blue outline), in original cm coordinates."""
    import json

    with open(spec_path) as f:
        spec = json.load(f)
    panel = spec["pattern"]["panels"][panel_name]
    vertices = np.array(panel["vertices"], dtype=np.float64)
    edges = panel["edges"]

    n = len(edge_indices)
    ncols = int(np.ceil(np.sqrt(n)))
    nrows = int(np.ceil(n / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4.5 * nrows), layout="constrained")
    axs = np.atleast_2d(axs).flatten()

    for idx, edge_idx in enumerate(edge_indices):
        ax = axs[idx]
        edge = edges[edge_idx]
        start = vertices[edge["endpoints"][0]]
        end = vertices[edge["endpoints"][1]]
        curvature = edge.get("curvature")

        true_pts = gt_sample_edge(start, end, curvature, 64)
        prims = _fit_single_edge(start, end, curvature, fit_tol_cm)

        margin = 0.2 * max(np.ptp(true_pts[:, 0]), np.ptp(true_pts[:, 1]), 1e-6)
        ax.set_xlim(true_pts[:, 0].min() - margin, true_pts[:, 0].max() + margin)
        ax.set_ylim(true_pts[:, 1].min() - margin, true_pts[:, 1].max() + margin)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(f"edge {edge_idx} ({len(prims)} prim)", fontsize=9)

        ax.plot(true_pts[:, 0], true_pts[:, 1], color=(0.89, 0.69, 0.73), linewidth=6, zorder=0)
        MPLVisualizerAxes(ax).visualize_shape(
            Shape(prims),
            line_style=LineStyle(color="blue", linewidth=1.2, arrowwidth=None),
            arc_style=LineStyle(color="blue", linewidth=1.2, arrowwidth=None),
            point_style=PointStyle(color="blue", radius=10),
        )

    for idx in range(n, len(axs)):
        axs[idx].axis("off")

    fig.suptitle(f"{panel_name}: true curve (pink, thick) vs. our fit (blue)")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")
