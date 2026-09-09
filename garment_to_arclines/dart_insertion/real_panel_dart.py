"""Phase E, Step 2: run the (already fully tested on synthetic panels)
dart-insertion transform on a REAL GarmentCodeData panel for the first
time -- rand_5DVS167AF5, right_ftorso (see real_panel.py for how its edges
were hand-identified).

apex and the side-seam dart-point fraction are hardcoded inputs, proposed
from the panel's own geometry (not derived by any rule): apex sits roughly
centered between center-front and the armhole/side-seam corner, a bit
above the dart point on the side seam, mirroring the synthetic panel's
apex placement convention.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from conversion.loader import load_panel  # noqa: E402
from dart_insertion.chains import sample_chain  # noqa: E402
from dart_insertion.transform import inches_to_t, insert_dart  # noqa: E402

SPEC_PATH = "../GarmentCodeData/garments_5000_0/default_body/data/rand_5DVS167AF5/rand_5DVS167AF5_specification.json"
PANEL_NAME = "right_ftorso"

# real panel's own edge_labels only carry GarmentCodeData's names
# (right_collar, right_armhole) and leave shoulder/side_seam/hem/center_front
# unlabeled -- relabel to match insert_dart's expected names, using the
# hand-identification from real_panel.py's Step 1. edge_map (and therefore
# the actual primitives) is untouched -- this only renames which edge
# indices insert_dart looks up by which name.
EDGE_RELABEL = {0: "center_front", 1: "collar", 2: "shoulder", 3: "armhole", 4: "side_seam", 5: "hem"}

APEX_RAW = (-11.0, 6.0)  # proposed: interior, centered-ish, above the dart point
SIDE_SEAM_T = 0.65  # proposed: 65% down the side seam from the armhole end


def _load_relabeled():
    data = load_panel(SPEC_PATH, PANEL_NAME)
    data.edge_labels = dict(EDGE_RELABEL)
    return data


def _draw_result(ax, panel, result, title: str, show_original: bool = True, show_joints: bool = True) -> None:
    if show_original:
        orig = np.array([p.numpy() for p in sample_chain(panel.shape.primitives)] + [panel.shape.primitives[0].start.numpy()])
        ax.plot(orig[:, 0], orig[:, 1], color="lightgray", linewidth=1.2, linestyle="--", zorder=1)
        if show_joints:
            orig_joints = np.array([p.start.numpy() for p in panel.shape.primitives])
            ax.scatter(orig_joints[:, 0], orig_joints[:, 1], color="lightgray", s=8, zorder=2)

    pts = np.array([p.numpy() for p in sample_chain(result.boundary)] + [result.boundary[0].start.numpy()])
    ax.plot(pts[:, 0], pts[:, 1], color="black", linewidth=1.5, zorder=3)

    if show_joints:
        # the REAL underlying primitive boundaries -- every individual
        # Line/Arc's own start point, not just the smooth sampled curve.
        # This is what actually makes up the armhole's 26-arc chain etc.
        joints = np.array([p.start.numpy() for p in result.boundary])
        ax.scatter(joints[:, 0], joints[:, 1], color="crimson", s=10, zorder=4)

    v0 = result.v0_original.detach().numpy()
    ax.scatter(*v0, color="gray", s=20, zorder=4, marker="x")
    ax.scatter(*result.apex.detach().numpy(), color="red", s=25, zorder=5)
    ax.scatter(*result.hinge.detach().numpy(), color="blue", s=25, zorder=5)
    ax.scatter(*result.tip.detach().numpy(), color="darkorange", s=25, zorder=5)

    ax.set_aspect("equal")
    ax.set_title(title, fontsize=9)


def render_before_after(out_path: str, inches: float = 0.5) -> None:
    panel = _load_relabeled()
    t = inches_to_t(panel, inches)
    result = insert_dart(panel, APEX_RAW, SIDE_SEAM_T, t)

    fig, axs = plt.subplots(1, 2, figsize=(11, 8))
    _draw_result(axs[0], panel, insert_dart(panel, APEX_RAW, SIDE_SEAM_T, 0.0), "before (spread=0)")
    _draw_result(
        axs[1],
        panel,
        result,
        f'after (spread={inches}")\ncf_extension={result.cf_extension:+.4f}, theta={torch.rad2deg(result.theta).item():.1f}deg',
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def render_sweep(out_path: str, inch_steps: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)) -> None:
    panel = _load_relabeled()
    fig, axs = plt.subplots(1, len(inch_steps), figsize=(3.4 * len(inch_steps), 7))
    for ax, inches in zip(axs, inch_steps):
        t = inches_to_t(panel, inches)
        result = insert_dart(panel, APEX_RAW, SIDE_SEAM_T, t)
        _draw_result(ax, panel, result, f'spread={inches}"\ncf_extension={result.cf_extension:+.4f}')
        dart_leg_gap = (result.rotated_dart_point - result.translated_dart_point).norm().item()
        print(
            f'spread={inches:.3f}"  t={t:.5f}  theta={torch.rad2deg(result.theta).item():7.3f}deg  '
            f"cf_extension={result.cf_extension:+.5f}  dart_leg_gap={dart_leg_gap:.5f}"
        )
    fig.suptitle("Phase E: real panel (rand_5DVS167AF5, right_ftorso) dart sweep")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def render_gif(out_path: str, max_inches: float = 1.0, n_frames: int = 30, fps: int = 12, hold_frames: int = 10) -> None:
    """Full-panel animation, spread 0 -> max_inches -> 0 (ping-pong loop),
    fixed window across every frame (computed from the union of all frames'
    boundaries, not just the last -- center-front and the hem both move)
    so only the geometry moves, not the view. Boundary is drawn from the
    real sampled arc chain (26-primitive armhole, single-arc collar), same
    as the static renders -- every arc primitive genuinely re-sampled each
    frame, not interpolated as an image."""
    import io

    from PIL import Image

    panel = _load_relabeled()
    inch_steps = list(np.linspace(0, max_inches, n_frames))
    results = [insert_dart(panel, APEX_RAW, SIDE_SEAM_T, inches_to_t(panel, inches)) for inches in inch_steps]

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
        _draw_result(ax, panel, result, f'spread = {inches:.3f}"')
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
    out_dir = Path(__file__).resolve().parent / "out" / "real_panel"
    out_dir.mkdir(parents=True, exist_ok=True)
    render_before_after(str(out_dir / "real_panel_before_after.png"))
    render_sweep(str(out_dir / "real_panel_sweep.png"))
    render_gif(str(out_dir / "real_panel_sweep.gif"))
