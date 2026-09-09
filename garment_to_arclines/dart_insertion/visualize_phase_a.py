"""Phase A: synthetic panel + apex/hinge/rays, visualized -- no dart
transform exists yet. Just verifying the geometric setup looks right."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dart_insertion.geometry import fraction_along, one_third_up, perpendicular_foot  # noqa: E402
from dart_insertion.synthetic_panel import build_synthetic_torso_panel, edge_by_label  # noqa: E402


def render_phase_a(out_path: str, apex_raw: tuple[float, float] = (-13.0, 29.0), side_seam_t: float = 0.25) -> None:
    panel = build_synthetic_torso_panel()

    # apex given in the panel's own raw (pre-normalization) cm coords, for
    # readability -- pushed through the same transform as the panel itself.
    apex = torch.tensor(panel.transform.apply(np.array(apex_raw)), dtype=torch.float32)

    armhole = edge_by_label(panel, "armhole")
    side_seam = edge_by_label(panel, "side_seam")
    hem = edge_by_label(panel, "hem")

    hinge = one_third_up(armhole)
    # side_seam.start is the armhole-end corner (see synthetic_panel vertex
    # order), so t=0 is the armhole end and t=1 is the hem end, matching the
    # agreed convention.
    dart_point = fraction_along(side_seam, side_seam_t)
    hem_foot = perpendicular_foot(apex, hem)

    fig, ax = plt.subplots(figsize=(6, 7))

    boundary = np.array([p.start.numpy() for p in panel.shape.primitives] + [panel.shape.primitives[0].start.numpy()])
    ax.plot(boundary[:, 0], boundary[:, 1], color="black", linewidth=1.5, zorder=1)

    for edge_idx, label in panel.edge_labels.items():
        prim = panel.shape.primitives[panel.edge_map[edge_idx][0]]
        mid = ((prim.start + prim.end) / 2).numpy()
        ax.annotate(label, mid, fontsize=8, color="gray", ha="center")

    points = [
        (apex, "apex", "red"),
        (hinge, "armhole hinge (1/3 up)", "blue"),
        (dart_point, f"line-3 endpoint (t={side_seam_t})", "green"),
        (hem_foot, "hem foot (perp.)", "purple"),
    ]
    for p, label, color in points:
        p_ = p.detach().numpy()
        ax.scatter(*p_, color=color, zorder=3, s=40)
        ax.annotate(label, p_, fontsize=8, color=color, xytext=(5, 5), textcoords="offset points")

    a = apex.detach().numpy()
    rays = [
        (hem_foot, "purple", "line 1: apex -> hem"),
        (hinge, "blue", "line 2: apex -> hinge"),
        (dart_point, "green", "line 3: apex -> side seam"),
    ]
    for target, color, _label in rays:
        t_ = target.detach().numpy()
        ax.plot([a[0], t_[0]], [a[1], t_[1]], color=color, linewidth=1, linestyle="--", zorder=2)

    ax.set_aspect("equal")
    ax.set_title("Phase A: synthetic panel, apex, hinge, rays (no dart yet)")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent / "out" / "phase_a"
    out_dir.mkdir(parents=True, exist_ok=True)
    render_phase_a(str(out_dir / "phase_a.png"))
