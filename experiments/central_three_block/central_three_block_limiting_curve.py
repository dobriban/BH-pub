#!/usr/bin/env python3
"""Deterministic quadrature for the central model's limiting FDR curve."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import time

import numpy as np

from central_three_block_curve_core import ALPHAS, batch_curve


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "reproduced" / "central_three_block"


def evaluate(
    dz: float = 0.001, zmax: float = 9.0, batch: int = 500
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    if dz <= 0 or zmax <= 0 or batch <= 0:
        raise ValueError("dz, zmax, and batch must be positive")
    z = np.arange(-zmax, zmax + dz / 2, dz, dtype=float)
    weights = np.exp(-z * z / 2) / math.sqrt(2 * math.pi) * dz
    weights[[0, -1]] *= 0.5
    total = np.zeros_like(ALPHAS)
    crossing = np.zeros_like(ALPHAS)
    maximum_c = np.zeros_like(ALPHAS)
    started = time.time()
    for start in range(0, len(z), batch):
        zz = z[start : start + batch]
        fdp, cutoffs, has_crossing = batch_curve(zz)
        ww = weights[start : start + batch, None]
        total += (ww * fdp).sum(axis=0)
        crossing += (ww * has_crossing).sum(axis=0)
        maximum_c = np.maximum(
            maximum_c, np.max(np.where(has_crossing, cutoffs, 0), axis=0)
        )
        if start and start % (batch * 10) == 0:
            print(
                "progress",
                start,
                "/",
                len(z),
                "elapsed",
                time.time() - started,
                flush=True,
            )
    return z, total, crossing, maximum_c, time.time() - started


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dz", type=float, default=0.001)
    parser.add_argument("--zmax", type=float, default=9.0)
    parser.add_argument("--batch", type=int, default=500)
    args = parser.parse_args()

    z, values, crossing, maximum_c, elapsed = evaluate(
        dz=args.dz, zmax=args.zmax, batch=args.batch
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "limiting_fdr_curve.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "alpha",
                "limiting_fdr",
                "excess",
                "ratio",
                "crossing_probability",
                "maximum_c_star",
            ]
        )
        for alpha, value, probability, cutoff in zip(
            ALPHAS, values, crossing, maximum_c
        ):
            writer.writerow(
                [
                    f"{alpha:.3f}",
                    f"{value:.12f}",
                    f"{value - alpha:.12f}",
                    f"{value / alpha:.9f}",
                    f"{probability:.12f}",
                    f"{cutoff:.9f}",
                ]
            )
    payload = {
        "mesh": {
            "z_min": float(z[0]),
            "z_max": float(z[-1]),
            "dz": float(z[1] - z[0]),
            "z_points": len(z),
        },
        "elapsed_seconds": elapsed,
        "rows": [
            {
                "alpha": float(alpha),
                "limiting_fdr": float(value),
                "excess": float(value - alpha),
                "ratio": float(value / alpha),
                "crossing_probability": float(probability),
                "maximum_c_star": float(cutoff),
            }
            for alpha, value, probability, cutoff in zip(
                ALPHAS, values, crossing, maximum_c
            )
        ],
    }
    json_path = args.output_dir / "limiting_fdr_curve.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    for index in [0, 1, 4, 9, 49, 99]:
        print(
            ALPHAS[index],
            values[index],
            values[index] - ALPHAS[index],
            crossing[index],
            maximum_c[index],
        )
    print("wrote", csv_path, "and", json_path, "elapsed", elapsed)


if __name__ == "__main__":
    main()
