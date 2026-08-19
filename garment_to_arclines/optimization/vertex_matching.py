"""Optimize a panel's D4D Shape parameters (every primitive's start/end, and
k for arcs) to match a genuinely different real target -- the same garment
design (same GarmentCodeData id), fit to a different body.

Both the source and target panel are loaded normally (via load_panel, each
independently fit/normalized), then the target's primitives are re-expressed
in the SOURCE's normalization frame (not independently re-normalized) --
both are affine (translate+scale) transforms of the same underlying raw cm
data, so converting is just: source_transform.apply(target_transform.invert(p)).
That keeps both shapes comparable in one consistent frame, so the
optimization has to correct the true relative size/shape difference between
the two body fits, not just match two independently size-normalized shapes.

Source and target are matched primitive-for-primitive (flat list, in order).
This requires:
  - matching *vertex* topology (same edge count/endpoint structure) between
    source and target -- true for the pairs used here, but not guaranteed in
    general (GarmentCode's stitch-matching subdivision can add a different
    number of vertices depending on the body, see rand_RGH9DN47T5).
  - matching *fitted primitive* counts per edge, for quadratic/cubic edges --
    the source and target curves are fit independently, so in general there's
    no guarantee they produce the same number of arcs. This only works when
    they happen to match (checked explicitly, not assumed); the two curves
    being similar in shape (same design, different body) makes this common
    but not guaranteed. Handling a mismatch is future work.
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from conversion.loader import DEFAULT_FIT_TOL_CM, load_panel  # noqa: E402

from d4descent.objects.arclines import Arc, Line, Primitive, Shape  # noqa: E402
from optimization.scale_optimization import _draw_snapshot, _shape_lim  # noqa: E402


def _snapshot_shape(shape: Shape) -> Shape:
    """Detached, cloned copy of shape -- for storing/rendering without
    holding onto the live autograd graph."""
    cache: dict[int, torch.Tensor] = {}

    def cloned(t: torch.Tensor) -> torch.Tensor:
        if id(t) not in cache:
            cache[id(t)] = t.detach().clone()
        return cache[id(t)]

    new_prims: list[Primitive] = []
    for p in shape.primitives:
        if isinstance(p, Arc):
            new_prims.append(Arc(cloned(p.start), cloned(p.end), p.k.detach().clone()))
        else:
            new_prims.append(Line(cloned(p.start), cloned(p.end)))
    return Shape(new_prims)


def _reexpress_in_source_frame(target_data, source_data) -> list[Primitive]:
    """Target's fitted primitives, converted from target's own normalized
    frame into source's normalized frame. Both frames are affine transforms
    of the same raw cm coordinates, so: apply(source, invert(target, p))."""
    k_scale = source_data.transform.scale / target_data.transform.scale
    point_cache: dict[int, torch.Tensor] = {}

    def convert_point(t: torch.Tensor) -> torch.Tensor:
        if id(t) not in point_cache:
            raw = target_data.transform.invert(t.detach().numpy())
            point_cache[id(t)] = torch.tensor(source_data.transform.apply(raw), dtype=torch.float32)
        return point_cache[id(t)]

    converted: list[Primitive] = []
    for p in target_data.shape.primitives:
        start, end = convert_point(p.start), convert_point(p.end)
        if isinstance(p, Arc):
            converted.append(Arc(start, end, p.k.detach() * k_scale))
        else:
            converted.append(Line(start, end))
    return converted


def run(
    source_spec: str,
    target_spec: str,
    panel_name: str,
    steps: int = 300,
    snapshot_every: int = 5,
    lr: float = 0.05,
    fit_tol_cm: float = DEFAULT_FIT_TOL_CM,
):
    source_data = load_panel(source_spec, panel_name, fit_tol_cm=fit_tol_cm)
    target_data = load_panel(target_spec, panel_name, fit_tol_cm=fit_tol_cm)
    source_shape = source_data.shape
    src_prims = source_shape.primitives
    target_prims = _reexpress_in_source_frame(target_data, source_data)

    if len(src_prims) != len(target_prims):
        raise ValueError(
            f"Primitive count mismatch for '{panel_name}': source has {len(src_prims)}, "
            f"target has {len(target_prims)}. Direct primitive-for-primitive matching "
            "requires equal counts (see module docstring)."
        )

    # optimizable params: every unique leaf tensor across source_shape's primitives
    unique_params: dict[int, torch.Tensor] = {}
    for p in src_prims:
        unique_params[id(p.start)] = p.start
        unique_params[id(p.end)] = p.end
        if isinstance(p, Arc):
            unique_params[id(p.k)] = p.k
    for t in unique_params.values():
        t.requires_grad_(True)
    params = list(unique_params.values())
    optimizer = torch.optim.Adam(params, lr=lr)

    with torch.no_grad():
        target_shape = Shape(target_prims)

    n_arcs = sum(1 for p in src_prims if isinstance(p, Arc))
    history = []
    snapshots: list[tuple[int, Shape]] = []
    for step in range(steps):
        if step % snapshot_every == 0:
            snapshots.append((step, _snapshot_shape(source_shape)))

        optimizer.zero_grad()
        loss = torch.tensor(0.0)
        for sp, tp in zip(src_prims, target_prims):
            loss = loss + ((sp.start - tp.start) ** 2).sum() + ((sp.end - tp.end) ** 2).sum()
            if isinstance(sp, Arc):
                loss = loss + (sp.k - tp.k) ** 2
        loss.backward()
        optimizer.step()
        history.append((step, loss.item()))

    snapshots.append((steps, _snapshot_shape(source_shape)))

    print(f"panel: {panel_name}  ({len(src_prims)} primitives, {n_arcs} arcs, {len(params)} free parameters)")
    for step, loss_val in history[:: max(steps // 10, 1)]:
        print(f"  step {step:4d}: loss={loss_val:.6f}")
    final_loss = history[-1][1]
    print(f"final loss = {final_loss:.6f}")
    assert final_loss < 1e-4, "panel did not converge to target"
    print("PASS: source panel converged to target panel")

    return history, target_shape, snapshots


def render_filmstrip(source_spec: str, target_spec: str, panel_name: str, out_path: str, n_frames: int = 10, **kwargs) -> None:
    import matplotlib.pyplot as plt

    _history, target_shape, snapshots = run(source_spec, target_spec, panel_name, **kwargs)

    idxs = np.linspace(0, len(snapshots) - 1, n_frames).round().astype(int)
    idxs = sorted(set(idxs.tolist()))
    chosen = [snapshots[i] for i in idxs]

    lim = _shape_lim([target_shape] + [s for _, s in chosen])
    fig, axs = plt.subplots(1, len(chosen), figsize=(2.6 * len(chosen), 3), layout="constrained")
    for ax, (step, shape) in zip(axs, chosen):
        _draw_snapshot(ax, shape, target_shape, lim, f"step {step}")
    fig.suptitle(f"{panel_name}: default_body (blue) optimizing toward random_body (green)")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def render_gif(source_spec: str, target_spec: str, panel_name: str, out_path: str, fps: int = 15, **kwargs) -> None:
    import io

    import matplotlib.pyplot as plt
    from PIL import Image

    _history, target_shape, snapshots = run(source_spec, target_spec, panel_name, **kwargs)
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
        append_images=frames[1:] + [frames[-1]] * fps,
        duration=1000 // fps,
        loop=0,
    )
    print(f"Wrote {out_path} ({len(frames)} frames)")


def _garment_specs(gid: str) -> tuple[str, str]:
    root = "../GarmentCodeData/garments_5000_0"
    source_spec = f"{root}/default_body/data/{gid}/{gid}_specification.json"
    target_spec = f"{root}/random_body/data/{gid}/{gid}_specification.json"
    return source_spec, target_spec


def _out_dir(gid: str, panel_name: str) -> Path:
    d = Path(__file__).resolve().parent / "out" / gid / panel_name
    d.mkdir(parents=True, exist_ok=True)
    return d


if __name__ == "__main__":
    for gid, panel_name in [
        ("rand_SM87ZO7LRL", "skirt_front"),  # straight edges only
        ("rand_UJYYSPSCTW", "wb_front"),  # circle edges
        ("rand_RYYOALW9XZ", "right_sleeve_f"),  # 1 cubic edge -> 79 fitted arcs
        ("rand_04ANOD2PBA", "right_btorso"),  # straight + quadratic (incl. 2 darts) + circle, 14 verts
    ]:
        source_spec, target_spec = _garment_specs(gid)
        out_dir = _out_dir(gid, panel_name)
        render_filmstrip(source_spec, target_spec, panel_name, str(out_dir / "filmstrip.png"))
        render_gif(source_spec, target_spec, panel_name, str(out_dir / "animation.gif"))
