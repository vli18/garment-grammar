"""Phase B: sanity-check the dart-angle equation and lower-region
translation on the synthetic panel's actual apex/hinge/hem/dart-point
geometry. No region splitting or rigid-motion application yet."""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dart_insertion.equations import angle_at_apex, dart_angle, lower_region_translation  # noqa: E402
from dart_insertion.geometry import fraction_along, one_third_up, perpendicular_foot  # noqa: E402
from dart_insertion.synthetic_panel import build_synthetic_torso_panel, edge_by_label  # noqa: E402


def main(apex_raw: tuple[float, float] = (-13.0, 29.0), side_seam_t: float = 0.25) -> None:
    panel = build_synthetic_torso_panel()
    apex = torch.tensor(panel.transform.apply(np.array(apex_raw)), dtype=torch.float32)

    armhole = edge_by_label(panel, "armhole")
    side_seam = edge_by_label(panel, "side_seam")
    hem = edge_by_label(panel, "hem")

    hinge = one_third_up(armhole)
    dart_point = fraction_along(side_seam, side_seam_t)
    hem_foot = perpendicular_foot(apex, hem)

    alpha = angle_at_apex(apex, hem_foot, hinge)
    a = (hinge - apex).norm()

    print(f"alpha = {torch.rad2deg(alpha).item():.2f} deg  ({alpha.item():.4f} rad)")
    print(f"a (|apex -> hinge|) = {a.item():.4f}")
    print()
    header = f"{'t':>6} | {'theta (deg)':>12} | {'translation (out, down)':>26}"
    print(header)
    print("-" * len(header))
    for t in [0.0, 0.02, 0.05, 0.08, 0.1]:
        theta = dart_angle(alpha, a, t)
        trans = lower_region_translation(apex, hem_foot, dart_point, alpha, theta, t)
        print(f"{t:6.3f} | {torch.rad2deg(theta).item():12.4f} | ({trans[0].item():+.5f}, {trans[1].item():+.5f})")


if __name__ == "__main__":
    main()
