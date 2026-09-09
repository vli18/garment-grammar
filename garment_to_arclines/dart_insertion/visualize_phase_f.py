"""Phase F: visual check that the AddDartSpec/apply_add_dart D4D-style
wrapper (rewrite.py) reproduces the same dart insertion as calling
insert_dart directly (Phase E). Forced-fire, fixed t, no candidate search --
this just confirms the rewrite packaging didn't change the geometry."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from conversion.loader import load_panel  # noqa: E402

from dart_insertion.chains import sample_chain  # noqa: E402
from dart_insertion.rewrite import (  # noqa: E402
    AddDartSpec,
    RemoveDartSpec,
    apply_add_dart,
    apply_remove_dart,
    generate_add_dart_specs,
)
from dart_insertion.transform import inches_to_t  # noqa: E402

SPEC_PATH = "../GarmentCodeData/garments_5000_0/default_body/data/rand_5DVS167AF5/rand_5DVS167AF5_specification.json"
PANEL_NAME = "right_ftorso"
EDGE_RELABEL = {0: "center_front", 1: "collar", 2: "shoulder", 3: "armhole", 4: "side_seam", 5: "hem"}
APEX_RAW = (-11.0, 6.0)
SIDE_SEAM_T = 0.65


def _load_relabeled():
    panel = load_panel(SPEC_PATH, PANEL_NAME)
    panel.edge_labels = dict(EDGE_RELABEL)
    return panel


def _draw_original(ax, panel) -> None:
    orig = np.array([p.numpy() for p in sample_chain(panel.shape.primitives)] + [panel.shape.primitives[0].start.numpy()])
    ax.plot(orig[:, 0], orig[:, 1], color="lightgray", linewidth=1.2, linestyle="--", zorder=1)


def _draw_joints(ax, boundary, color: str = "crimson") -> None:
    """Every individual primitive's own start point -- not just the smooth
    sampled curve. This is what actually shows a new vertex the moment it's
    introduced (e.g. the dart point split out of the side seam, or the tip/
    new-corner points), even while it's sitting exactly on top of another
    point (t=0, nothing has moved apart yet)."""
    joints = np.array([p.start.numpy() for p in boundary])
    ax.scatter(joints[:, 0], joints[:, 1], color=color, s=14, zorder=4)


def _draw(ax, panel, result, title: str) -> None:
    _draw_original(ax, panel)

    pts = np.array([p.numpy() for p in sample_chain(result.boundary)] + [result.boundary[0].start.numpy()])
    ax.plot(pts[:, 0], pts[:, 1], color="black", linewidth=1.5, zorder=3)
    _draw_joints(ax, result.boundary)

    ax.scatter(*result.apex.detach().numpy(), color="red", s=25, zorder=5, label="apex")
    ax.scatter(*result.hinge.detach().numpy(), color="blue", s=25, zorder=5, label="hinge")
    ax.scatter(*result.tip.detach().numpy(), color="darkorange", s=25, zorder=5, label="tip")
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=9)


def _draw_boundary(ax, panel, boundary, title: str) -> None:
    """Like _draw, but for a plain (dart-free) boundary -- no
    apex/hinge/tip markers, since a reconstructed panel has no dart on it
    anymore. Drawn as a real solid outline (not a dashed diagnostic overlay)
    so it reads as an actual panel, just colored to distinguish it as the
    reconstructed one."""
    _draw_original(ax, panel)

    pts = np.array([p.numpy() for p in sample_chain(boundary)] + [boundary[0].start.numpy()])
    ax.plot(pts[:, 0], pts[:, 1], color="forestgreen", linewidth=2, zorder=3)
    _draw_joints(ax, boundary, color="darkgreen")
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=9)


def render_forced_fire(out_path: str, inches: float = 0.5) -> None:
    panel = _load_relabeled()

    # generate -> apply -> apply (reverse): the actual pipeline as built,
    # not just insert_dart called directly.
    specs = generate_add_dart_specs(panel, APEX_RAW, SIDE_SEAM_T, t_inches=inches)
    assert len(specs) == 1, "panel should be eligible for AddDart"
    spec = specs[0]

    unfired = apply_add_dart(panel, AddDartSpec(apex_raw=spec.apex_raw, side_seam_t=spec.side_seam_t, t=0.0))
    fired = apply_add_dart(panel, spec)
    reversed_ = apply_remove_dart(fired, RemoveDartSpec())

    fig, axs = plt.subplots(1, 3, figsize=(16, 8))
    _draw(axs[0], panel, unfired, "generate + AddDartSpec(t=0)\n-- rewrite not fired")
    _draw(axs[1], panel, fired, f'apply_add_dart\n-- forced fire, spread={inches}"')
    axs[1].legend(fontsize=7, loc="upper right")

    # reversed_.boundary is a real re-assembled closed boundary (not just
    # loose points) -- drawn as its own polygon, dashed green on top of the
    # original gray dashed outline, so an exact overlap is visually obvious.
    _draw_boundary(axs[2], panel, reversed_.boundary, "apply_remove_dart\n-- reconstructed panel vs. original")

    fig.suptitle(
        "Phase F: insert_dart wrapped as a D4D-style rewrite pair\n"
        "(apex/side_seam_t fixed, t forced -- green = apply_remove_dart's reconstructed boundary)"
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def render_gif(
    out_path: str, max_inches: float = 0.5, n_frames: int = 24, fps: int = 12, hold_frames: int = 8
) -> None:
    """Animates the full Phase F pipeline: apply_add_dart sweeping the
    dart open then closed (continuous, t: 0 -> max -> 0, via
    generate_add_dart_specs each frame -- Jump Continuity in action), then
    a distinct held segment showing apply_remove_dart's actual reconstructed
    boundary (green), confirming the discrete reverse rewrite lands on the
    same place the continuous sweep already visits at t=0."""
    import io

    from PIL import Image

    panel = _load_relabeled()
    inch_steps = list(np.linspace(0, max_inches, n_frames))
    open_results = []
    for inches in inch_steps:
        specs = generate_add_dart_specs(panel, APEX_RAW, SIDE_SEAM_T, t_inches=inches)
        open_results.append(apply_add_dart(panel, specs[0]))
    close_results = list(reversed(open_results))

    fired_max = open_results[-1]
    reversed_ = apply_remove_dart(fired_max, RemoveDartSpec())

    all_pts = np.concatenate(
        [np.array([p.numpy() for p in sample_chain(r.boundary)]) for r in open_results]
        + [np.array([p.numpy() for p in sample_chain(panel.shape.primitives)])]
    )
    lo, hi = all_pts.min(axis=0), all_pts.max(axis=0)
    margin = 0.08 * max((hi - lo).max(), 0.1)
    xlim = (lo[0] - margin, hi[0] + margin)
    ylim = (lo[1] - margin, hi[1] + margin)

    def _frame(draw_fn, *args) -> Image.Image:
        fig, ax = plt.subplots(figsize=(6, 7))
        draw_fn(ax, panel, *args)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110)
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
        plt.close(fig)
        return img

    frames = []
    for inches, r in zip(inch_steps, open_results):
        frames.append(_frame(_draw, r, f'apply_add_dart, spread={inches:.3f}"'))
    frames += [frames[-1]] * hold_frames
    for inches, r in zip(reversed(inch_steps), close_results):
        frames.append(_frame(_draw, r, f'apply_add_dart, spread={inches:.3f}"'))
    frames += [frames[-1]] * hold_frames
    frames += [_frame(_draw_boundary, reversed_.boundary, "apply_remove_dart -- reconstructed boundary")] * hold_frames

    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=1000 // fps,
        loop=0,
    )
    print(f"Wrote {out_path} ({len(frames)} frames)")


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent / "out" / "phase_f"
    out_dir.mkdir(parents=True, exist_ok=True)
    render_forced_fire(str(out_dir / "phase_f_forced_fire.png"))
    render_gif(str(out_dir / "phase_f_sweep.gif"))
