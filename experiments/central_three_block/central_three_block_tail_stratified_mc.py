#!/usr/bin/env python3
"""Probability-stratified Monte Carlo for the central limiting FDR curve."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np
from scipy.special import ndtri
from scipy.stats import t as student_t

from central_three_block_curve_core import ALPHAS, batch_curve


SEGMENTS = [(0.0, 1.0, 1200)]
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "reproduced" / "central_three_block"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--quadrature", type=Path)
    parser.add_argument("--macro-replications", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260822)
    args = parser.parse_args()
    if args.macro_replications < 2:
        parser.error("--macro-replications must be at least 2")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    quadrature_path = args.quadrature or args.output_dir / "limiting_fdr_curve.csv"
    if not quadrature_path.exists():
        parser.error(
            f"quadrature file not found: {quadrature_path}; run the limiting curve first"
        )

    rng = np.random.default_rng(args.seed)
    macros = np.empty((args.macro_replications, len(ALPHAS)))
    started = time.time()
    for replication in range(args.macro_replications):
        estimate = np.zeros(len(ALPHAS))
        for u_lower, u_upper, strata in SEGMENTS:
            u = u_lower + (u_upper - u_lower) * (
                np.arange(strata) + rng.random(strata)
            ) / strata
            z = ndtri(u)
            fdp, _, _ = batch_curve(z)
            estimate += (u_upper - u_lower) * fdp.mean(axis=0)
        macros[replication] = estimate
        print(
            "macro",
            replication + 1,
            "/",
            args.macro_replications,
            "elapsed",
            time.time() - started,
            flush=True,
        )

    mean = macros.mean(axis=0)
    standard_deviation = macros.std(axis=0, ddof=1)
    mcse = standard_deviation / np.sqrt(args.macro_replications)
    critical = student_t.ppf(0.975, args.macro_replications - 1)
    lower = mean - critical * mcse
    upper = mean + critical * mcse
    quadrature = np.genfromtxt(quadrature_path, delimiter=",", names=True)

    macros_path = args.output_dir / "tail_stratified_mc_macros.csv"
    with macros_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["macro_replication"] + [f"alpha_{alpha:.3f}" for alpha in ALPHAS]
        )
        for index, row in enumerate(macros, start=1):
            writer.writerow([index] + [f"{value:.12f}" for value in row])

    curve_path = args.output_dir / "tail_stratified_mc_curve.csv"
    with curve_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "alpha",
                "mc_estimate",
                "mcse",
                "ci95_lo",
                "ci95_hi",
                "deterministic_quadrature",
                "estimate_minus_alpha",
            ]
        )
        for alpha, value, standard_error, low, high, quad in zip(
            ALPHAS,
            mean,
            mcse,
            lower,
            upper,
            quadrature["limiting_fdr"],
        ):
            writer.writerow(
                [
                    f"{alpha:.3f}",
                    f"{value:.12f}",
                    f"{standard_error:.12f}",
                    f"{low:.12f}",
                    f"{high:.12f}",
                    f"{quad:.12f}",
                    f"{value - alpha:.12f}",
                ]
            )
    payload = {
        "design": {
            "segments": [
                {"u_lo": lower, "u_hi": upper, "strata": strata}
                for lower, upper, strata in SEGMENTS
            ],
            "total_strata_per_macro": sum(strata for _, _, strata in SEGMENTS),
            "macro_replications": args.macro_replications,
            "seed": args.seed,
            "alpha_grid": ["0.001", "0.100", "0.001"],
        },
        "elapsed_seconds": time.time() - started,
        "rows": [
            {
                "alpha": float(alpha),
                "estimate": float(value),
                "mcse": float(standard_error),
                "ci95": [float(low), float(high)],
                "deterministic_quadrature": float(quad),
            }
            for alpha, value, standard_error, low, high, quad in zip(
                ALPHAS,
                mean,
                mcse,
                lower,
                upper,
                quadrature["limiting_fdr"],
            )
        ],
    }
    json_path = args.output_dir / "tail_stratified_mc_curve.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    for index in [0, 1, 4, 9, 49, 99]:
        print(
            ALPHAS[index],
            mean[index],
            mcse[index],
            lower[index],
            upper[index],
            quadrature["limiting_fdr"][index],
        )
    print("wrote", curve_path, macros_path, "and", json_path)
    print("elapsed", time.time() - started)


if __name__ == "__main__":
    main()
