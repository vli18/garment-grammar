"""Gradient sanity check: load one panel, mark its primitive parameters as
optimizable, run one trivial loss + backward pass, confirm gradients are
non-zero.

This is the actual point of the whole conversion effort -- a Shape that
loads and visualizes correctly but can't be differentiated through isn't
useful for gradient-based optimization.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from conversion.loader import load_panel  # noqa: E402

from d4descent.objects.arclines import Arc  # noqa: E402


def gradient_sanity_check(spec_path: str, panel_name: str) -> None:
    data = load_panel(spec_path, panel_name)
    prims = data.shape.primitives

    # Enable grad on every unique leaf tensor (start/end are shared across
    # primitives at joints, so dedupe by identity before enabling/counting).
    leaves = {}
    for p in prims:
        leaves[id(p.start)] = p.start
        leaves[id(p.end)] = p.end
        if isinstance(p, Arc):
            leaves[id(p.k)] = p.k
    for t in leaves.values():
        t.requires_grad_(True)

    # Trivial loss, but one that actually exercises D4D's own Arc geometry
    # (compute_o_r_thetas / sample), not just a pass-through sum.
    loss = torch.tensor(0.0)
    for p in prims:
        loss = loss + (p.start**2).sum() + (p.end**2).sum()
        if isinstance(p, Arc):
            loss = loss + p.k**2
            mid = p.sample(torch.tensor(0.5))
            loss = loss + (mid**2).sum()

    loss.backward()

    print(f"panel: {panel_name}  ({len(prims)} primitives, {len(leaves)} unique leaf tensors)")
    print(f"loss = {loss.item():.6f}")
    n_nonzero = 0
    n_none = 0
    for t in leaves.values():
        if t.grad is None:
            n_none += 1
        elif torch.all(t.grad == 0):
            pass
        else:
            n_nonzero += 1
    print(f"leaves with non-zero grad: {n_nonzero} / {len(leaves)}  (None: {n_none})")
    assert n_none == 0, "some leaves got no gradient at all"
    assert n_nonzero == len(leaves), "some leaves got an all-zero gradient"
    print("PASS: all leaf tensors received non-zero gradients")


if __name__ == "__main__":
    gradient_sanity_check(
        "../GarmentCodeData/rand_2M8H83CQJP/rand_2M8H83CQJP_specification.json",
        "wb_front",
    )
