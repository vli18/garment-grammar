"""Phase E, Step 1: load one real GarmentCodeData panel and hand-identify
its armhole/side-seam/hem edges, hardcoded for this specific panel -- no
automatic edge-label detection yet.

Panel: rand_5DVS167AF5, right_ftorso (default_body). DARTLESS -- checked
directly against the data (no self-stitched edge pairs) -- and structurally
identical to our synthetic panel (6 edges: center_front, collar, shoulder,
armhole, side_seam, hem, each a single edge, no dart splitting side_seam or
hem). Real curvature on both collar (circle) and armhole (cubic), both
converge cleanly (31 primitives total, no max-depth/non-convergence
warnings -- unlike several other candidates checked that hit pathological
non-converging cubic curves).
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from conversion.loader import PanelData, load_panel  # noqa: E402
from d4descent.objects.arclines import Arc, Line  # noqa: E402

SPEC_PATH = "../GarmentCodeData/garments_5000_0/default_body/data/rand_5DVS167AF5/rand_5DVS167AF5_specification.json"
PANEL_NAME = "right_ftorso"

# hand-identified from specification.json + stitches (see conversation) --
# original GarmentCodeData edge indices, grouped into our named edges.
EDGE_GROUPS = {
    "center_front": [0],
    "collar": [1],
    "shoulder": [2],
    "armhole": [3],
    "side_seam": [4],
    "hem": [5],
}

COLORS = {
    "center_front": "gray",
    "collar": "purple",
    "shoulder": "gray",
    "armhole": "blue",
    "side_seam": "green",
    "hem": "darkorange",
}


def primitives_for_edges(data: PanelData, edge_indices: list[int]) -> list[Line | Arc]:
    prims = []
    for e in edge_indices:
        prims.extend(data.shape.primitives[i] for i in data.edge_map[e])
    return prims


def sample_prims(prims: list[Line | Arc], n_per_arc: int = 15) -> np.ndarray:
    pts = []
    for p in prims:
        if isinstance(p, Arc):
            import torch

            pts.extend(p.sample(torch.linspace(0, 1, n_per_arc)).numpy())
        else:
            pts.append(p.start.numpy())
    pts.append(prims[-1].end.numpy())
    return np.array(pts)


def render_labeled_panel(out_path: str) -> None:
    data = load_panel(SPEC_PATH, PANEL_NAME)

    fig, ax = plt.subplots(figsize=(6, 8))
    for name, edge_idxs in EDGE_GROUPS.items():
        prims = primitives_for_edges(data, edge_idxs)
        pts = sample_prims(prims)
        color = COLORS[name]
        lw = 1.8
        ax.plot(pts[:, 0], pts[:, 1], color=color, linewidth=lw, label=f"{name} (edges {edge_idxs})")
        mid = pts[len(pts) // 2]
        ax.annotate(name, mid, fontsize=7, color=color, xytext=(4, 4), textcoords="offset points")

    ax.set_aspect("equal")
    ax.set_title(f"{PANEL_NAME} (rand_5DVS167AF5, default_body)\nhand-identified edges (dartless)")
    ax.legend(fontsize=6, loc="upper right")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")

    print()
    print("Primitive counts per named edge:")
    for name, edge_idxs in EDGE_GROUPS.items():
        prims = primitives_for_edges(data, edge_idxs)
        n_arc = sum(1 for p in prims if isinstance(p, Arc))
        n_line = sum(1 for p in prims if isinstance(p, Line))
        print(f"  {name:14s} edges={edge_idxs}  {len(prims)} primitive(s)  ({n_arc} arc, {n_line} line)")


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent / "out" / "real_panel"
    out_dir.mkdir(parents=True, exist_ok=True)
    render_labeled_panel(str(out_dir / "real_panel_labeled.png"))
