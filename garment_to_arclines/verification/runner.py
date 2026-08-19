
"""Generic verification runner for any GarmentCodeData garment folder:
round-trip check, whole-garment side-by-side, and per-panel overlay.
"""

import json
import sys
from pathlib import Path

import numpy as np

_PKG_ROOT = Path(__file__).resolve().parent.parent
_REPO_ROOT = _PKG_ROOT.parent
sys.path.insert(0, str(_PKG_ROOT))
# Absolute (not relative) imports throughout this file: it's normally run
# directly as a script (`python verification/runner.py ...`), which gives it
# no package context for relative imports to resolve against.
from conversion.loader import load_panel  # noqa: E402
from verification.compare import render_comparison, render_overlay  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "out"


def check_round_trip(spec_path: str, panel_name: str) -> None:
    with open(spec_path) as f:
        spec = json.load(f)
    orig_vertices = np.array(spec["pattern"]["panels"][panel_name]["vertices"], dtype=np.float64)

    data = load_panel(spec_path, panel_name)

    recovered_normalized = np.array(
        [data.shape.primitives[data.edge_map[i][0]].start.numpy() for i in sorted(data.edge_map)],
        dtype=np.float64,
    )
    recovered = data.transform.invert(recovered_normalized)

    max_err = np.abs(recovered - orig_vertices).max()
    status = "OK" if max_err < 1e-4 else "FAIL"
    print(f"[{status}] {panel_name}: round-trip max abs error = {max_err:.3e} cm")


def verify_garment(garment_name: str, panel_names: list[str]) -> None:
    garment_dir = _REPO_ROOT / "GarmentCodeData" / garment_name
    spec_path = str(garment_dir / f"{garment_name}_specification.json")
    reference_png = str(garment_dir / f"{garment_name}_pattern.png")

    for panel_name in panel_names:
        check_round_trip(spec_path, panel_name)

    out_dir = OUT_DIR / garment_name
    out_dir.mkdir(parents=True, exist_ok=True)
    render_comparison(spec_path, panel_names, reference_png, str(out_dir / "comparison.png"))
    render_overlay(spec_path, panel_names, str(out_dir / "overlay.png"))


if __name__ == "__main__":
    import argparse

    from verification.garments import GARMENTS

    parser = argparse.ArgumentParser(description="Verify GarmentCodeData panels load correctly into D4D Shapes.")
    parser.add_argument("garment", help="GarmentCodeData folder name, e.g. rand_2E2EL4UZUS")
    parser.add_argument("--panels", nargs="+", default=None, help="Override the registered panel list")
    args = parser.parse_args()

    panel_names = args.panels or GARMENTS.get(args.garment)
    if panel_names is None:
        raise SystemExit(
            f"No registered panel list for '{args.garment}' in garments.py; pass --panels explicitly."
        )

    verify_garment(args.garment, panel_names)
