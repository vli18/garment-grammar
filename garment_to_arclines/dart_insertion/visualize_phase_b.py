"""Phase B visual check: overlay, on the Phase A panel/apex/hinge/rays,
where the rotation (theta, about the hinge) and the lower-region
translation actually send the apex, for a few t. Still no region
splitting/reassembly -- this is just the two equations, previewed on the
real geometry instead of read off a table of numbers."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dart_insertion.equations import angle_at_apex, dart_angle, lower_region_translation  # noqa: E402
from dart_insertion.geometry import fraction_along, one_third_up, perpendicular_foot, rotate_about  # noqa: E402
from dart_insertion.synthetic_panel import build_synthetic_torso_panel, edge_by_label  # noqa: E402


def render_phase_b(
    out_path: str,
    apex_raw: tuple[float, float] = (-13.0, 29.0),
    side_seam_t: float = 0.25,
    ts: tuple[float, ...] = (0.02, 0.05, 0.08, 0.1),
) -> None:
    panel = build_synthetic_torso_panel()
    apex = torch.tensor(panel.transform.apply(np.array(apex_raw)), dtype=torch.float32)

    armhole = edge_by_label(panel, "armhole")
    side_seam = edge_by_label(panel, "side_seam")
    hem = edge_by_label(panel, "hem")

    hinge = one_third_up(armhole)
    dart_point = fraction_along(side_seam, side_seam_t)
    hem_foot = perpendicular_foot(apex, hem)
    alpha = angle_at_apex(apex, hem_foot, hinge)
    a = (hinge - apex).norm()

    fig, ax = plt.subplots(figsize=(6, 7))

    boundary = np.array([p.start.numpy() for p in panel.shape.primitives] + [panel.shape.primitives[0].start.numpy()])
    ax.plot(boundary[:, 0], boundary[:, 1], color="black", linewidth=1.5, zorder=1)
    for edge_idx, label in panel.edge_labels.items():
        prim = panel.shape.primitives[panel.edge_map[edge_idx][0]]
        mid = ((prim.start + prim.end) / 2).numpy()
        ax.annotate(label, mid, fontsize=8, color="lightgray", ha="center")

    # original (t=0) reference geometry, dim
    for p, color in [(apex, "red"), (hinge, "blue"), (dart_point, "green"), (hem_foot, "purple")]:
        ax.scatter(*p.detach().numpy(), color=color, zorder=3, s=30, alpha=0.4)
    a_ = apex.detach().numpy()
    for target, color in [(hem_foot, "purple"), (hinge, "blue"), (dart_point, "green")]:
        t_ = target.detach().numpy()
        ax.plot([a_[0], t_[0]], [a_[1], t_[1]], color=color, linewidth=1, linestyle="--", zorder=2, alpha=0.4)
    ax.annotate("apex (t=0)", a_, fontsize=8, color="red", xytext=(5, -10), textcoords="offset points")
    ax.annotate("hinge", hinge.detach().numpy(), fontsize=8, color="blue", xytext=(5, 5), textcoords="offset points")

    cmap = plt.get_cmap("plasma")
    for i, t in enumerate(ts):
        theta = dart_angle(alpha, a, t)
        trans = lower_region_translation(apex, hem_foot, dart_point, alpha, theta, t)
        color = cmap(0.15 + 0.7 * i / max(len(ts) - 1, 1))

        # upper-side region: apex AND dart_point both rotate about the hinge
        rotated_apex = rotate_about(apex, hinge, theta)
        rotated_dart_point = rotate_about(dart_point, hinge, theta)

        # lower-side region: apex, hem_foot, AND dart_point all translate
        translated_apex = apex + trans
        translated_hem_foot = hem_foot + trans
        translated_dart_point = dart_point + trans

        h_ = hinge.detach().numpy()
        ra_ = rotated_apex.detach().numpy()
        rd_ = rotated_dart_point.detach().numpy()
        ta_ = translated_apex.detach().numpy()
        th_ = translated_hem_foot.detach().numpy()
        td_ = translated_dart_point.detach().numpy()

        # upper-side preview: new line 2 (hinge->apex) and new line 3 (apex->dart_point)
        ax.plot([h_[0], ra_[0]], [h_[1], ra_[1]], color=color, linewidth=1.3, zorder=4)
        ax.plot([ra_[0], rd_[0]], [ra_[1], rd_[1]], color=color, linewidth=1.3, linestyle="-.", zorder=4)
        ax.scatter(*ra_, color=color, marker="^", s=35, zorder=5)
        ax.scatter(*rd_, color=color, marker="^", s=25, zorder=5)

        # lower-side preview: new line 1 (apex->hem_foot) and new line 3 (apex->dart_point)
        ax.plot([ta_[0], th_[0]], [ta_[1], th_[1]], color=color, linewidth=1.3, linestyle=":", zorder=4)
        ax.plot([ta_[0], td_[0]], [ta_[1], td_[1]], color=color, linewidth=1.3, linestyle="--", zorder=4)
        ax.scatter(*ta_, color=color, marker="s", s=35, zorder=5)
        ax.scatter(*th_, color=color, marker="s", s=25, zorder=5)
        ax.scatter(*td_, color=color, marker="s", s=25, zorder=5)
        ax.annotate(f"t={t}", ta_, fontsize=7, color=color, xytext=(5, -3), textcoords="offset points")

    from matplotlib.lines import Line2D

    legend_elems = [
        Line2D([0], [0], color="gray", linestyle="-", label="upper-side new line 2 (hinge->apex')"),
        Line2D([0], [0], color="gray", linestyle="-.", label="upper-side new line 3 (apex'->dart_point')"),
        Line2D([0], [0], color="gray", linestyle=":", label="lower-side new line 1 (apex'->hem_foot')"),
        Line2D([0], [0], color="gray", linestyle="--", label="lower-side new line 3 (apex'->dart_point')"),
    ]
    ax.legend(handles=legend_elems, fontsize=7, loc="lower right")

    ax.set_aspect("equal")
    ax.set_title(f"Phase B: rotation + translation previews vs t\n(alpha={torch.rad2deg(alpha).item():.1f} deg)")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent / "out" / "phase_b"
    out_dir.mkdir(parents=True, exist_ok=True)
    render_phase_b(str(out_dir / "phase_b.png"))
