"""Standalone edge-level checks used during development of quadratic/cubic
support -- kept as reproducible scripts rather than one-off commands.

1. right_btorso's quadratic edges, checked individually (at the time these
   were added, right_btorso also had cubic edges not yet supported, so the
   whole panel couldn't load -- this checks quadratic in isolation).
2. A synthetic, deliberately dramatic quadratic curve, to confirm adaptive
   subdivision actually subdivides (real GarmentCode quadratic edges in the
   example data are all nearly symmetric/circular, so they don't exercise
   this on their own).
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from conversion.arc_fit import fit_curve_to_arcs  # noqa: E402
from conversion.bezier import sample_quadratic  # noqa: E402
from d4descent.objects.arclines import Shape  # noqa: E402
from d4descent.visualizer import MPLVisualizer  # noqa: E402
from verification.compare import render_edge_overlay  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "out" / "edge_checks"


def right_btorso_quadratic_edges() -> None:
    spec = "../GarmentCodeData/rand_2M8H83CQJP/rand_2M8H83CQJP_specification.json"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    render_edge_overlay(spec, "right_btorso", [0, 3, 6], str(OUT_DIR / "right_btorso_quadratic_edges.png"))


def synthetic_quadratic_subdivision() -> None:
    """A dramatic, asymmetric bulge (real edges in the example data are all
    near-symmetric) at a tight tolerance, to confirm adaptive subdivision
    actually produces multiple smoothly-joined arcs, not just one."""
    start = np.array([0.0, 0.0])
    end = np.array([10.0, 0.0])
    control_rel = np.array([0.5, 0.6])

    def sample_fn(t):
        return sample_quadratic(start, end, control_rel, t)

    start_t = torch.tensor(start, dtype=torch.float32)
    end_t = torch.tensor(end, dtype=torch.float32)
    tol = 0.02
    prims = fit_curve_to_arcs(sample_fn, start_t, end_t, tol)
    print(f"synthetic_quadratic_subdivision: {len(prims)} primitives for tol={tol}")
    for i in range(len(prims) - 1):
        assert prims[i].end is prims[i + 1].start, f"chain broken at primitive {i}"

    shape = Shape(prims)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig = MPLVisualizer(1, 1, 8, 4, xlim=(-1, 11), ylim=(-2, 8), notebook=False)
    fig[0].visualize_shape(shape, show_text=True)
    import matplotlib.pyplot as plt

    plt.imsave(str(OUT_DIR / "synthetic_quadratic_subdivision.png"), fig.get_image())
    print(f"Wrote {OUT_DIR / 'synthetic_quadratic_subdivision.png'}")


if __name__ == "__main__":
    right_btorso_quadratic_edges()
    synthetic_quadratic_subdivision()
