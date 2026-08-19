"""First real (if toy) optimization test: reparameterize a loaded panel by a
single global scale factor and confirm gradient descent recovers a known
target scale.

Unlike gradient_sanity.py (checks *some* gradient flows), this checks the
gradient is *correct*: we know the exact answer (target_scale), so we can
verify convergence, not just non-zero-ness.

Scaling is done in normalized coordinates, where load_panel already centers
the panel at the origin -- so "scale about the panel's own center" is simply
`scale * point` (and `scale * k` for arcs, since k is a length and must scale
the same way as the chord it belongs to, or the curve would distort).
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from conversion.loader import load_panel  # noqa: E402

from d4descent.objects.arclines import Arc, Line, Primitive, Shape  # noqa: E402


def build_scaled_shape(shape: Shape, scale: torch.Tensor) -> Shape:
    """New Shape where every primitive is `shape`'s geometry scaled by `scale`
    about the origin. Preserves shared-vertex connectivity."""
    scaled_cache: dict[int, torch.Tensor] = {}

    def scaled(t: torch.Tensor) -> torch.Tensor:
        if id(t) not in scaled_cache:
            scaled_cache[id(t)] = scale * t
        return scaled_cache[id(t)]

    new_prims: list[Primitive] = []
    for p in shape.primitives:
        if isinstance(p, Arc):
            new_prims.append(Arc(scaled(p.start), scaled(p.end), scale * p.k))
        else:
            new_prims.append(Line(scaled(p.start), scaled(p.end)))
    return Shape(new_prims)


def run(
    spec_path: str,
    panel_name: str,
    target_scale: float = 1.3,
    init_scale: float = 1.0,
    steps: int = 200,
    snapshot_every: int = 5,
) -> None:
    data = load_panel(spec_path, panel_name)
    orig_shape = data.shape

    # Target: what the panel should look like at target_scale (fixed, no grad).
    with torch.no_grad():
        target_points = []
        for p in orig_shape.primitives:
            target_points.append(target_scale * p.start)
            if isinstance(p, Arc):
                target_points.append(torch.tensor([target_scale * p.k, 0.0]))  # pack k alongside points

    scale = torch.tensor(init_scale, requires_grad=True)
    optimizer = torch.optim.Adam([scale], lr=0.05)

    history = []
    snapshots: list[tuple[int, Shape]] = []
    for step in range(steps):
        if step % snapshot_every == 0:
            with torch.no_grad():
                snapshots.append((step, build_scaled_shape(orig_shape, scale.detach().clone())))

        optimizer.zero_grad()
        scaled_shape = build_scaled_shape(orig_shape, scale)

        loss = torch.tensor(0.0)
        idx = 0
        for p in scaled_shape.primitives:
            loss = loss + ((p.start - target_points[idx]) ** 2).sum()
            idx += 1
            if isinstance(p, Arc):
                loss = loss + (p.k - target_points[idx][0]) ** 2
                idx += 1

        loss.backward()
        optimizer.step()
        history.append((step, scale.item(), loss.item()))

    with torch.no_grad():
        snapshots.append((steps, build_scaled_shape(orig_shape, scale.detach().clone())))

    print(f"panel: {panel_name}  target_scale={target_scale}  init_scale={init_scale}")
    for step, s, loss_val in history[::steps // 10]:
        print(f"  step {step:4d}: scale={s:.5f}  loss={loss_val:.6f}")
    final_scale = history[-1][1]
    print(f"final scale = {final_scale:.5f} (target {target_scale})")
    assert abs(final_scale - target_scale) < 1e-2, "scale did not converge to target"
    print("PASS: scale converged to target")

    with torch.no_grad():
        initial_shape = build_scaled_shape(orig_shape, torch.tensor(init_scale))
        final_shape = build_scaled_shape(orig_shape, torch.tensor(final_scale))
        target_shape = build_scaled_shape(orig_shape, torch.tensor(target_scale))

    return history, initial_shape, final_shape, target_shape, snapshots


def render(spec_path: str, panel_name: str, out_path: str, **kwargs) -> None:
    import matplotlib.pyplot as plt

    from d4descent.visualizer import LineStyle, MPLVisualizerAxes, PointStyle

    history, initial_shape, final_shape, target_shape, _snapshots = run(spec_path, panel_name, **kwargs)
    steps_, scales_, _losses = zip(*history)

    fig, axs = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")

    ax = axs[0]
    all_pts = torch.stack([p.start for p in target_shape.primitives] + [p.start for p in initial_shape.primitives])
    lim = float(all_pts.abs().max()) * 1.2
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"{panel_name}: initial (gray) / target (green, dashed look) / optimized (blue)")
    vis = MPLVisualizerAxes(ax)
    vis.visualize_shape(initial_shape, line_style=LineStyle("lightgray", 2, None), arc_style=LineStyle("lightgray", 2, None), point_style=PointStyle("lightgray", 4))
    vis.visualize_shape(target_shape, line_style=LineStyle("green", 4, None), arc_style=LineStyle("green", 4, None), point_style=PointStyle("green", 6))
    vis.visualize_shape(final_shape, line_style=LineStyle("blue", 1.5, None), arc_style=LineStyle("blue", 1.5, None), point_style=PointStyle("blue", 8))

    ax2 = axs[1]
    ax2.plot(steps_, scales_, color="blue")
    ax2.axhline(scales_[0] * 0 + kwargs.get("target_scale", 1.3), color="green", linestyle="--", label="target")
    ax2.set_xlabel("optimization step")
    ax2.set_ylabel("scale")
    ax2.set_title("scale convergence")
    ax2.legend()

    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def _shape_lim(shapes: list[Shape]) -> float:
    all_pts = torch.stack([p.start for shape in shapes for p in shape.primitives])
    return float(all_pts.abs().max()) * 1.2


def _draw_snapshot(ax, shape: Shape, target_shape: Shape, lim: float, title: str) -> None:
    from d4descent.visualizer import LineStyle, MPLVisualizerAxes, PointStyle

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=9)
    vis = MPLVisualizerAxes(ax)
    vis.visualize_shape(
        target_shape,
        line_style=LineStyle("lightgreen", 3, None),
        arc_style=LineStyle("lightgreen", 3, None),
        point_style=PointStyle("lightgreen", 4),
    )
    vis.visualize_shape(
        shape,
        line_style=LineStyle("blue", 1.5, None),
        arc_style=LineStyle("blue", 1.5, None),
        point_style=PointStyle("blue", 6),
    )


def render_filmstrip(spec_path: str, panel_name: str, out_path: str, n_frames: int = 10, **kwargs) -> None:
    """Static grid: shape at evenly-spaced steps (blue) against the fixed
    target (green), so convergence over the course of optimization is
    visible in one image."""
    import matplotlib.pyplot as plt

    _history, _initial, _final, target_shape, snapshots = run(spec_path, panel_name, **kwargs)

    # pick n_frames evenly spaced snapshots, always including the first and last
    idxs = np.linspace(0, len(snapshots) - 1, n_frames).round().astype(int)
    idxs = sorted(set(idxs.tolist()))
    chosen = [snapshots[i] for i in idxs]

    lim = _shape_lim([target_shape] + [s for _, s in chosen])
    ncols = len(chosen)
    fig, axs = plt.subplots(1, ncols, figsize=(2.6 * ncols, 3), layout="constrained")
    for ax, (step, shape) in zip(axs, chosen):
        _draw_snapshot(ax, shape, target_shape, lim, f"step {step}")
    fig.suptitle(f"{panel_name}: optimization progress (blue) vs. target (green)")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def render_gif(spec_path: str, panel_name: str, out_path: str, fps: int = 15, **kwargs) -> None:
    """Animated GIF of the shape converging toward the target over the
    course of optimization."""
    import io

    import matplotlib.pyplot as plt
    from PIL import Image

    _history, _initial, _final, target_shape, snapshots = run(spec_path, panel_name, **kwargs)
    lim = _shape_lim([target_shape] + [s for _, s in snapshots])

    frames = []
    for step, shape in snapshots:
        fig, ax = plt.subplots(figsize=(4, 4), layout="constrained")
        _draw_snapshot(ax, shape, target_shape, lim, f"{panel_name}  step {step}")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100)
        buf.seek(0)
        frames.append(Image.open(buf).convert("RGB"))
        plt.close(fig)

    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:] + [frames[-1]] * (fps),  # hold last frame ~1s
        duration=1000 // fps,
        loop=0,
    )
    print(f"Wrote {out_path} ({len(frames)} frames)")


if __name__ == "__main__":
    spec = "../GarmentCodeData/rand_2E2EL4UZUS/rand_2E2EL4UZUS_specification.json"
    out_dir = Path(__file__).resolve().parent / "out" / "rand_2E2EL4UZUS" / "skirt_front"
    out_dir.mkdir(parents=True, exist_ok=True)
    render(spec, "skirt_front", str(out_dir / "convergence.png"), target_scale=1.3)
    render_filmstrip(spec, "skirt_front", str(out_dir / "filmstrip.png"), target_scale=1.3)
    render_gif(spec, "skirt_front", str(out_dir / "animation.gif"), target_scale=1.3)
