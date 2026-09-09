"""Phase D: before/after + a sweep of t, for the dart transform applied to
the CURVED (arc-chain) panel -- same rendering style as visualize_phase_c,
updated for boundary now being a real primitive chain (arcs included) and
for DartResult's named fields instead of positional boundary indices.

t values are picked in REAL units (inches of spread at line 1) via
transform.inches_to_t -- a realistic dart spread is well under 1", so
sweeping normalized t up to 0.6 (several inches) was unrealistic. Since
that means the visible motion is small relative to the whole panel, `zoom`
crops the view to the apex/hinge/dart region so it's actually visible.
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


def _zoom_bounds(result, margin: float = 0.6):
    pts = np.array(
        [
            result.apex.detach().numpy(),
            result.hinge.detach().numpy(),
            result.tip.detach().numpy(),
            result.rotated_dart_point.detach().numpy(),
            result.translated_dart_point.detach().numpy(),
        ]
    )
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    span = max((hi - lo).max(), 0.05)
    center = (lo + hi) / 2
    half = span / 2 + margin * span
    return (center[0] - half, center[0] + half), (center[1] - half, center[1] + half)


def _draw_result(
    ax, panel, result, title: str, show_original: bool = True, zoom: bool = False, show_rays: bool = False
) -> None:
    if show_original:
        orig = np.array([p.numpy() for p in sample_chain(panel.shape.primitives)] + [panel.shape.primitives[0].start.numpy()])
        ax.plot(orig[:, 0], orig[:, 1], color="lightgray", linewidth=1.2, linestyle="--", zorder=1)

    pts = np.array([p.numpy() for p in sample_chain(result.boundary)] + [result.boundary[0].start.numpy()])
    ax.plot(pts[:, 0], pts[:, 1], color="black", linewidth=1.5, zorder=3)

    # original V0, for reference -- shows how far center-front extended/shortened
    v0 = result.v0_original.detach().numpy()
    ax.scatter(*v0, color="gray", s=20, zorder=4, marker="x")

    apex = result.apex.detach().numpy()
    ax.scatter(*apex, color="red", s=25, zorder=5)
    ax.scatter(*result.hinge.detach().numpy(), color="blue", s=25, zorder=5)
    ax.scatter(*result.tip.detach().numpy(), color="darkorange", s=25, zorder=5)

    if show_rays:
        hinge = result.hinge.detach().numpy()
        hem_foot = result.hem_foot.detach().numpy()
        dart_point = result.dart_point.detach().numpy()
        rotated_dp = result.rotated_dart_point.detach().numpy()
        translated_dp = result.translated_dart_point.detach().numpy()
        apex_prime = result.apex_prime.detach().numpy()

        # line 1 (apex -> hem_foot): fixed reference, unchanged by t.
        ax.plot([apex[0], hem_foot[0]], [apex[1], hem_foot[1]], color="purple", linewidth=1, linestyle="--", zorder=2)
        ax.scatter(*hem_foot, color="purple", s=18, zorder=4)

        # line 2, ORIGINAL (apex -> hinge, dashed, static) vs. ROTATED
        # (hinge -> apex_prime, solid) -- this is "the line opening": at
        # t=0 apex_prime sits exactly on apex and the solid line lies on
        # top of the dashed one; as t grows apex_prime swings away from
        # apex about the hinge and the solid line visibly peels off.
        ax.plot([apex[0], hinge[0]], [apex[1], hinge[1]], color="steelblue", linewidth=1, linestyle="--", zorder=2)
        ax.plot([hinge[0], apex_prime[0]], [hinge[1], apex_prime[1]], color="steelblue", linewidth=1.6, zorder=4)

        # original (t=0) line 3 -- apex -> dart_point, before the spread opened it
        ax.plot([apex[0], dart_point[0]], [apex[1], dart_point[1]], color="lightgreen", linewidth=1, linestyle="--", zorder=2)
        ax.scatter(*dart_point, color="lightgreen", s=18, zorder=4)

        # the NATURAL (unadjusted) dart wedge: both legs go all the way to
        # apex_prime, not the practical drafted tip -- the angle at
        # apex_prime between these two legs is exactly theta (verified in
        # tests), unlike the tip-based wedge actually drawn into the panel
        # boundary above, which is pulled back for real-world drafting.
        ax.plot([rotated_dp[0], apex_prime[0]], [rotated_dp[1], apex_prime[1]], color="green", linewidth=1.6, zorder=4)
        ax.plot([apex_prime[0], translated_dp[0]], [apex_prime[1], translated_dp[1]], color="green", linewidth=1.6, zorder=4)
        ax.scatter(*apex_prime, color="green", s=22, zorder=5)

    ax.set_aspect("equal")
    ax.set_title(title, fontsize=9)
    if zoom:
        xlim, ylim = _zoom_bounds(result)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)


def render_before_after(out_path: str, inches: float = 0.75) -> None:
    panel = build_synthetic_torso_panel(curved=True)
    t = inches_to_t(panel, inches)
    result = insert_dart(panel, APEX_RAW, SIDE_SEAM_T, t)

    fig, axs = plt.subplots(1, 2, figsize=(11, 7))
    _draw_result(axs[0], panel, insert_dart(panel, APEX_RAW, SIDE_SEAM_T, 0.0), "before (spread=0)")
    _draw_result(
        axs[1],
        panel,
        result,
        f'after (spread={inches}")\ncf_extension={result.cf_extension:+.4f}, theta={torch.rad2deg(result.theta).item():.1f}deg',
        zoom=True,
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def render_sweep(out_path: str, inch_steps: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)) -> None:
    panel = build_synthetic_torso_panel(curved=True)
    fig, axs = plt.subplots(1, len(inch_steps), figsize=(3.2 * len(inch_steps), 6))
    for ax, inches in zip(axs, inch_steps):
        t = inches_to_t(panel, inches)
        result = insert_dart(panel, APEX_RAW, SIDE_SEAM_T, t)
        _draw_result(ax, panel, result, f'spread={inches}"\ncf_extension={result.cf_extension:+.4f}', zoom=True)
        dart_leg_gap = (result.rotated_dart_point - result.translated_dart_point).norm().item()
        print(
            f'spread={inches:.3f}"  t={t:.5f}  theta={torch.rad2deg(result.theta).item():7.3f}deg  '
            f"cf_extension={result.cf_extension:+.5f}  dart_leg_gap={dart_leg_gap:.5f}"
        )
    fig.suptitle('Phase D sweep: realistic spread in inches (zoomed to the dart region)')
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def render_gif(out_path: str, max_inches: float = 1.0, n_frames: int = 30, fps: int = 12, hold_frames: int = 10) -> None:
    """Animated GIF, spread going 0 -> max_inches -> 0 (so it loops
    cleanly), zoomed to a window fixed across every frame (computed from
    the final, most-open frame) so the view doesn't jitter as the dart
    grows -- only the geometry moves."""
    import io

    from PIL import Image

    panel = build_synthetic_torso_panel(curved=True)
    inch_steps = list(np.linspace(0, max_inches, n_frames))
    results = [insert_dart(panel, APEX_RAW, SIDE_SEAM_T, inches_to_t(panel, inches)) for inches in inch_steps]

    xlim, ylim = _zoom_bounds(results[-1])

    frames = []
    for inches, result in zip(inch_steps, results):
        fig, ax = plt.subplots(figsize=(6, 6))
        _draw_result(ax, panel, result, f'spread = {inches:.3f}"', zoom=False)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110)
        buf.seek(0)
        frames.append(Image.open(buf).convert("RGB"))
        plt.close(fig)

    # ping-pong (0 -> max -> 0) so the loop has no jump cut, hold a beat at each end
    sequence = frames + [frames[-1]] * hold_frames + frames[::-1] + [frames[0]] * hold_frames
    sequence[0].save(
        out_path,
        save_all=True,
        append_images=sequence[1:],
        duration=1000 // fps,
        loop=0,
    )
    print(f"Wrote {out_path} ({len(sequence)} frames)")


def render_gif_full(
    out_path: str,
    max_inches: float = 1.0,
    n_frames: int = 30,
    fps: int = 12,
    hold_frames: int = 10,
    show_rays: bool = False,
) -> None:
    """Same animation as render_gif, but showing the WHOLE panel (not
    zoomed to the dart region) -- so the hem/bottom seam sliding and
    center-front stretching are visible too, not just the notch itself.
    show_rays=True overlays line 1 (apex->hem_foot), line 2 (apex->hinge),
    the original (t=0) line 3 (apex->dart_point), and highlights the
    current dart wedge's two legs in green -- same construction lines
    visualize_phase_a/b showed, now on the live animated panel."""
    import io

    from PIL import Image

    panel = build_synthetic_torso_panel(curved=True)
    inch_steps = list(np.linspace(0, max_inches, n_frames))
    results = [insert_dart(panel, APEX_RAW, SIDE_SEAM_T, inches_to_t(panel, inches)) for inches in inch_steps]

    # fixed window covering every frame's full boundary (not just the last
    # one -- center-front's length and the hem's position both change over
    # the sweep, so the extremes aren't all in the final frame alone).
    all_pts = np.concatenate(
        [np.array([p.numpy() for p in sample_chain(r.boundary)]) for r in results]
        + [np.array([p.numpy() for p in sample_chain(panel.shape.primitives)])]
    )
    lo, hi = all_pts.min(axis=0), all_pts.max(axis=0)
    margin = 0.08 * max((hi - lo).max(), 0.1)
    xlim = (lo[0] - margin, hi[0] + margin)
    ylim = (lo[1] - margin, hi[1] + margin)

    frames = []
    for inches, result in zip(inch_steps, results):
        fig, ax = plt.subplots(figsize=(6, 7))
        _draw_result(ax, panel, result, f'spread = {inches:.3f}"', zoom=False, show_rays=show_rays)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110)
        buf.seek(0)
        frames.append(Image.open(buf).convert("RGB"))
        plt.close(fig)

    sequence = frames + [frames[-1]] * hold_frames + frames[::-1] + [frames[0]] * hold_frames
    sequence[0].save(
        out_path,
        save_all=True,
        append_images=sequence[1:],
        duration=1000 // fps,
        loop=0,
    )
    print(f"Wrote {out_path} ({len(sequence)} frames)")


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent / "out" / "phase_d"
    out_dir.mkdir(parents=True, exist_ok=True)
    render_before_after(str(out_dir / "phase_d_dart_before_after.png"))
    render_sweep(str(out_dir / "phase_d_dart_sweep.png"))
    render_gif(str(out_dir / "phase_d_dart_sweep.gif"))
    render_gif_full(str(out_dir / "phase_d_dart_sweep_full.gif"))
    render_gif_full(str(out_dir / "phase_d_dart_sweep_full_rays.gif"), show_rays=True)
