#!/usr/bin/env python3
"""High-precision direct Monte Carlo at alpha=0.01, 0.05, and 0.10.

This driver uses the exact conditional count-thinning simulator from
central_three_block_finite_sample_mc.py and reproduces the selected-level table
in the paper.  Each (N, alpha) pair receives an independent random-number
stream.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from central_three_block_finite_sample_mc import run_task

ALPHAS = (0.01, 0.05, 0.10)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--replications", type=int, default=500_000)
    parser.add_argument("--batch", type=int, default=20_000)
    parser.add_argument("--Ns", type=int, nargs="+", default=[50, 100, 500, 1000])
    parser.add_argument("--seed", type=int, default=20260921)
    args = parser.parse_args()
    if args.replications < 2:
        parser.error("--replications must be at least 2")
    if args.batch <= 0:
        parser.error("--batch must be positive")
    if any(N <= 0 for N in args.Ns):
        parser.error("all --Ns values must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    started = time.time()
    for N in args.Ns:
        for j, alpha in enumerate(ALPHAS):
            seed = args.seed + 100 * N + j
            row = run_task(
                (N, j, alpha, args.replications, args.batch, seed)
            )
            rows.append(row)
            print(
                f"N={N:4d} alpha={alpha:.2f} "
                f"FDR={row['fdr']:.12f} MCSE={row['mcse']:.3e}",
                flush=True,
            )

    rows.sort(key=lambda row: (row["N"], row["alpha"]))
    csv_path = args.output_dir / "finite_sample_selected_levels.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "model": {
            "block_sizes_per_N": [163, 1, 3],
            "means": ["0", "37/20", "59/12"],
            "loadings": ["3/10", "2/11", "20/21"],
            "residual_sds": [
                "sqrt(91)/10",
                "3*sqrt(13)/11",
                "sqrt(41)/21",
            ],
        },
        "method": "exact conditional binomial thinning to the greatest BH fixed point",
        "alphas": list(ALPHAS),
        "Ns": args.Ns,
        "replications_per_N_alpha": args.replications,
        "batch": args.batch,
        "base_seed": args.seed,
        "elapsed_seconds": time.time() - started,
        "rows": rows,
    }
    json_path = args.output_dir / "finite_sample_selected_levels.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {csv_path} and {json_path}", flush=True)


if __name__ == "__main__":
    main()
