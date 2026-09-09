"""Phase D, Step 1: the synthetic panel with real arc chains (collar,
armhole) instead of single straight Lines -- confirming the chains look
right before anything about the dart transform touches them."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from d4descent.objects.arclines import Arc  # noqa: E402

from dart_insertion.synthetic_panel import build_synthetic_torso_panel, edges_by_label  # noqa: E402


def render_phase_d_panel(out_path: str, n_samples_per_arc: int = 20) -> None:
    panel = build_synthetic_torso_panel(curved=True)

    fig, ax = plt.subplots(figsize=(6, 7))

    for edge_idx, label in panel.edge_labels.items():
        prims = edges_by_label(panel, label)
        color = "black"
        # smooth curve for plotting (samples within each primitive)
        pts = []
        for prim in prims:
            if isinstance(prim, Arc):
                ts = torch.linspace(0, 1, n_samples_per_arc)
                pts.extend(prim.sample(ts).numpy())
            else:
                pts.append(prim.start.numpy())
        pts.append(prims[-1].end.numpy())
        pts = np.array(pts)
        ax.plot(pts[:, 0], pts[:, 1], color=color, linewidth=1.5, zorder=2)

        # mark every primitive joint (the actual fitted vertices) along the chain
        joints = [prims[0].start.numpy()] + [p.end.numpy() for p in prims]
        joints = np.array(joints)
        marker_color = "crimson" if len(prims) > 1 else "gray"
        ax.scatter(joints[:, 0], joints[:, 1], color=marker_color, s=18, zorder=3)

        mid = pts[len(pts) // 2]
        ax.annotate(f"{label} ({len(prims)} prim)", mid, fontsize=8, color="dimgray", xytext=(4, 4), textcoords="offset points")

    ax.set_aspect("equal")
    ax.set_title("Phase D Step 1: synthetic panel with collar + armhole arc chains\n(red dots = fitted primitive joints)")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")

    for edge_idx, label in panel.edge_labels.items():
        prims = edges_by_label(panel, label)
        n_arcs = sum(1 for p in prims if isinstance(p, Arc))
        print(f"{label:20s} {len(prims)} primitive(s)  ({n_arcs} arc, {len(prims) - n_arcs} line)")


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent / "out" / "phase_d"
    out_dir.mkdir(parents=True, exist_ok=True)
    render_phase_d_panel(str(out_dir / "phase_d_panel.png"))
