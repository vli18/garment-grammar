"""Phase C: before/after + a sweep of t, for the actual assembled
dart-inserted boundary. Original V0 is shown dim/gray for reference so the
center-front extension is visible."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dart_insertion.synthetic_panel import build_synthetic_torso_panel  # noqa: E402
from dart_insertion.transform import insert_dart  # noqa: E402

APEX_RAW = (-13.0, 29.0)
SIDE_SEAM_T = 0.25


def _draw_result(ax, panel, result, title: str, show_original: bool = True) -> None:
    if show_original:
        orig = np.array(
            [p.start.numpy() for p in panel.shape.primitives] + [panel.shape.primitives[0].start.numpy()]
        )
        ax.plot(orig[:, 0], orig[:, 1], color="lightgray", linewidth=1.2, linestyle="--", zorder=1)

    pts = [p.detach().numpy() for p in result.boundary] + [result.boundary[0].detach().numpy()]
    pts = np.array(pts)
    ax.plot(pts[:, 0], pts[:, 1], color="black", linewidth=1.5, zorder=3)

    # original V0, for reference -- shows how far center-front extended/shortened
    v0 = result.v0_original.detach().numpy()
    ax.scatter(*v0, color="gray", s=20, zorder=4, marker="x")

    ax.scatter(*result.apex.detach().numpy(), color="red", s=25, zorder=5)
    ax.scatter(*result.hinge.detach().numpy(), color="blue", s=25, zorder=5)
    ax.scatter(*result.tip.detach().numpy(), color="darkorange", s=25, zorder=5)

    ax.set_aspect("equal")
    ax.set_title(title, fontsize=9)


def render_before_after(out_path: str, t: float = 0.08) -> None:
    panel = build_synthetic_torso_panel()
    result = insert_dart(panel, APEX_RAW, SIDE_SEAM_T, t)

    fig, axs = plt.subplots(1, 2, figsize=(11, 7))
    _draw_result(axs[0], panel, insert_dart(panel, APEX_RAW, SIDE_SEAM_T, 0.0), "before (t=0)")
    _draw_result(
        axs[1],
        panel,
        result,
        f"after (t={t})\ncf_extension={result.cf_extension:+.4f}, theta={torch.rad2deg(result.theta).item():.1f}deg",
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def render_sweep(out_path: str, ts: tuple[float, ...] = (0.0, 0.03, 0.06, 0.09)) -> None:
    panel = build_synthetic_torso_panel()
    fig, axs = plt.subplots(1, len(ts), figsize=(3.2 * len(ts), 6))
    for ax, t in zip(axs, ts):
        result = insert_dart(panel, APEX_RAW, SIDE_SEAM_T, t)
        _draw_result(ax, panel, result, f"t={t}\ncf_extension={result.cf_extension:+.4f}")
        print(
            f"t={t:.3f}  theta={torch.rad2deg(result.theta).item():7.3f}deg  "
            f"cf_extension={result.cf_extension:+.5f}  dart_leg_gap={(result.boundary[6]-result.boundary[8]).norm().item():.5f}"
        )
    fig.suptitle("Phase C sweep: assembled boundary (gray x = original V0, for reference)")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent / "out" / "phase_c"
    out_dir.mkdir(parents=True, exist_ok=True)
    render_before_after(str(out_dir / "phase_c_before_after.png"))
    render_sweep(str(out_dir / "phase_c_sweep.png"))
