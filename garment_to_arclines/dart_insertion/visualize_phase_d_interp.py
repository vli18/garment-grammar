"""Phase D interpolation check: like visualize_phase_b's overlay (multiple
t's constructions drawn on one axes, color-coded), but using the CURVED
panel's actual rotated armhole chain (arcs included) instead of just a
straight line2/line3 preview.

t steps are picked in REAL units (inches of spread at line 1), not
arbitrary normalized fractions -- a realistic dart spread is well under 1",
so the previous 0.05-0.65 normalized sweep was showing spreads of several
inches, wildly larger than anything a real dart would need. 1/8" steps up
to 1" covers the realistic range with the progression still visible.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dart_insertion.chains import sample_chain  # noqa: E402
from dart_insertion.synthetic_panel import build_synthetic_torso_panel  # noqa: E402
from dart_insertion.transform import inches_to_t, insert_dart  # noqa: E402

APEX_RAW = (-13.0, 29.0)
SIDE_SEAM_T = 0.25


def render_interpolation(out_path: str, inch_steps: tuple[float, ...] = tuple(np.arange(1, 9) / 8)) -> None:
    panel = build_synthetic_torso_panel(curved=True)
    ts = [inches_to_t(panel, inches) for inches in inch_steps]

    fig, ax = plt.subplots(figsize=(7, 8))

    orig = np.array([p.numpy() for p in sample_chain(panel.shape.primitives)] + [panel.shape.primitives[0].start.numpy()])
    ax.plot(orig[:, 0], orig[:, 1], color="lightgray", linewidth=1.2, linestyle="--", zorder=1)

    r0 = insert_dart(panel, APEX_RAW, SIDE_SEAM_T, 0.0)
    apex = r0.apex.detach().numpy()
    hinge = r0.hinge.detach().numpy()
    ax.scatter(*apex, color="red", s=30, zorder=6)
    ax.annotate("apex", apex, fontsize=8, color="red", xytext=(5, -10), textcoords="offset points")
    ax.scatter(*hinge, color="blue", s=30, zorder=6)
    ax.annotate("hinge", hinge, fontsize=8, color="blue", xytext=(5, 5), textcoords="offset points")

    cmap = plt.get_cmap("plasma")
    for i, (inches, t) in enumerate(zip(inch_steps, ts)):
        r = insert_dart(panel, APEX_RAW, SIDE_SEAM_T, t)
        color = cmap(0.1 + 0.8 * i / max(len(ts) - 1, 1))

        # upper-side: rotated armhole chain (hinge -> rotated_V3), curved, as one rigid piece
        upper_pts = np.array([p.numpy() for p in sample_chain(r.armhole_upper_side_rotated)] + [r.rotated_V3.detach().numpy()])
        ax.plot(upper_pts[:, 0], upper_pts[:, 1], color=color, linewidth=1.3, zorder=3)

        rv3 = r.rotated_V3.detach().numpy()
        rdp = r.rotated_dart_point.detach().numpy()
        tip = r.tip.detach().numpy()
        tdp = r.translated_dart_point.detach().numpy()
        tv4 = r.translated_V4.detach().numpy()
        thf = r.translated_hem_foot.detach().numpy()

        # upper side-seam portion (rotated_V3 -> rotated_dart_point)
        ax.plot([rv3[0], rdp[0]], [rv3[1], rdp[1]], color=color, linewidth=1.3, zorder=3)
        # dart legs
        ax.plot([rdp[0], tip[0]], [rdp[1], tip[1]], color=color, linewidth=1.3, linestyle="-.", zorder=3)
        ax.plot([tip[0], tdp[0]], [tip[1], tdp[1]], color=color, linewidth=1.3, linestyle="-.", zorder=3)
        # lower side-seam + hem (translated)
        ax.plot([tdp[0], tv4[0]], [tdp[1], tv4[1]], color=color, linewidth=1.3, linestyle=":", zorder=3)
        ax.plot([tv4[0], thf[0]], [tv4[1], thf[1]], color=color, linewidth=1.3, linestyle=":", zorder=3)

        ax.annotate(f'{inches:.3f}"', tdp, fontsize=6.5, color=color, xytext=(4, -4), textcoords="offset points")

    ax.set_aspect("equal")
    ax.set_title(
        "Phase D interpolation: rotated armhole chain + dart legs vs spread (in realistic inches)\n"
        "(solid=armhole+upper side-seam, -.=dart legs, :=lower side-seam+hem)"
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent / "out" / "phase_d"
    out_dir.mkdir(parents=True, exist_ok=True)
    render_interpolation(str(out_dir / "phase_d_interp.png"))
