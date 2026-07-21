#!/usr/bin/env python3
"""Certify the proof's FDR lower bound on alpha=0.001,...,0.100.

The floating-point calculation in ``generate_candidate_brackets`` only
proposes rational crossing witnesses.  The imported Arb checker independently
verifies every witness and every inequality used in the final bound.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import math
from decimal import Decimal
from pathlib import Path

import numpy as np
from flint import arb, fmpq
from scipy.special import log_ndtr, ndtri

from central_three_block_certificate import Certificate, SET


ALPHA_INDICES = np.arange(1, 101, dtype=int)
ALPHAS = ALPHA_INDICES.astype(float) / 1000
W = np.array([163 / 167, 1 / 167, 3 / 167], dtype=float)
MU = np.array([0, 37 / 20, 59 / 12], dtype=float)
RHO = np.array([3 / 10, 2 / 11, 20 / 21], dtype=float)
SD = np.sqrt(1 - RHO * RHO)
Z_INDICES = np.arange(-600, 600, dtype=int)
WITNESS_DENOMINATOR = 100_000_000
WITNESS_PADDING = 501


def alpha_string(index: int) -> str:
    return f"{index / 1000:.3f}"


def log_q(c: np.ndarray, a: np.ndarray, s: float) -> np.ndarray:
    return np.logaddexp(log_ndtr((a - c) / s), log_ndtr((-a - c) / s))


def lower_mean_envelopes() -> np.ndarray:
    """Return m^-_{g,k} as doubles for non-rigorous witness generation."""
    zl = Z_INDICES / 100
    zh = (Z_INDICES + 1) / 100
    out = np.empty((len(Z_INDICES), 3), dtype=float)
    for g, (mu, rho) in enumerate(zip(MU, RHO)):
        left = mu + rho * zl
        right = mu + rho * zh
        crosses_zero = np.minimum(left, right) <= 0
        crosses_zero &= 0 <= np.maximum(left, right)
        out[:, g] = np.where(
            crosses_zero, 0.0, np.minimum(np.abs(left), np.abs(right))
        )
    return out


def lower_envelope_log_mixture(c: np.ndarray, means: np.ndarray) -> np.ndarray:
    """Log of the three-block lower-envelope tail mixture."""
    value = None
    for g in range(3):
        term = math.log(W[g]) + log_q(c, means[..., g], SD[g])
        value = term if value is None else np.logaddexp(value, term)
    return value


def generate_candidate_brackets() -> dict[int, dict]:
    """Generate untrusted rational feasible witnesses for all grid levels."""
    means = lower_mean_envelopes()
    scan = np.arange(1.60, 20.0001, 0.01, dtype=float)
    scan_matrix = scan[None, :]
    mean_matrix = means[:, None, :]
    log_mix = lower_envelope_log_mixture(scan_matrix, mean_matrix)
    log_u = math.log(2) + log_ndtr(-scan)
    # Feasibility requires both u(c) <= alpha and alpha G(c)/u(c) >= 1.
    log_required_alpha = np.maximum(log_u[None, :], log_u[None, :] - log_mix)

    indices = np.empty((len(Z_INDICES), len(ALPHAS)), dtype=int)
    for column, alpha in enumerate(ALPHAS):
        feasible = log_required_alpha <= math.log(alpha)
        if not np.all(np.any(feasible, axis=1)):
            missing = Z_INDICES[~np.any(feasible, axis=1)]
            raise RuntimeError(f"candidate scan found no crossing in bins {missing}")
        indices[:, column] = np.argmax(feasible, axis=1)
    if np.any(indices == 0):
        raise RuntimeError("candidate scan begins above a crossing")

    hi = scan[indices]
    lo = scan[indices - 1]
    c_alpha = ndtri(1 - ALPHAS / 2)
    lo = np.maximum(lo, c_alpha[None, :])
    log_alpha = np.log(ALPHAS)[None, :]

    def score(c: np.ndarray) -> np.ndarray:
        return (
            log_alpha
            + lower_envelope_log_mixture(c, mean_matrix)
            - (math.log(2) + log_ndtr(-c))
        )

    if np.any(score(lo) > 0) or np.any(score(hi) < 0):
        raise RuntimeError("candidate scan did not bracket the first crossing")
    for _ in range(44):
        mid = (lo + hi) / 2
        is_feasible = score(mid) >= 0
        hi = np.where(is_feasible, mid, hi)
        lo = np.where(is_feasible, lo, mid)

    roots = (lo + hi) / 2
    a_num = np.floor(WITNESS_DENOMINATOR * roots).astype(np.int64)
    b_num = a_num + WITNESS_PADDING
    brackets = {}
    for column, index in enumerate(ALPHA_INDICES):
        brackets[int(index)] = {
            "alpha": alpha_string(int(index)),
            "z_denominator": 100,
            "bracket_denominator": WITNESS_DENOMINATOR,
            "rows": [
                {
                    "z_index": int(k),
                    "a_num": int(a),
                    "b_num": int(b),
                }
                for k, a, b in zip(
                    Z_INDICES, a_num[:, column], b_num[:, column]
                )
            ],
        }
    return brackets


def check_archived_brackets(base: Path, brackets: dict[int, dict]) -> None:
    """Require exact agreement with the three archived theorem-level files."""
    for index, archived_alpha in ((10, "0.01"), (50, "0.05"), (100, "0.10")):
        path = base / f"brackets_alpha_{archived_alpha}.json"
        archived = json.loads(path.read_text())
        if archived["bracket_denominator"] != WITNESS_DENOMINATOR:
            raise RuntimeError(f"unexpected witness denominator in {path}")
        if archived["rows"] != brackets[index]["rows"]:
            raise RuntimeError(f"generated witnesses do not reproduce {path}")


def certified_decimal_numerator(value: arb, places: int) -> int:
    """Largest n certified to satisfy n/10^places < value."""
    scale = 10**places
    numerator = math.floor(float(value) * scale)
    while not value > arb(fmpq(numerator, scale)):
        numerator -= 1
    while value > arb(fmpq(numerator + 1, scale)):
        numerator += 1
    return numerator


def fixed_decimal(numerator: int, places: int) -> str:
    return format(Decimal(numerator) / Decimal(10**places), f".{places}f")


def certify_grid(brackets: dict[int, dict], places: int) -> list[dict]:
    rows = []
    shared_mode_cache = {}
    for index in ALPHA_INDICES:
        index = int(index)
        alpha = alpha_string(index)
        c_alpha = float(ndtri(1 - index / 2000))
        cstart_num = math.floor(1000 * c_alpha)
        SET[alpha] = {
            "zden": 100,
            "zlo": -600,
            "zhi": 600,
            "cstart": fmpq(cstart_num, 1000),
            # This exact comparison also certifies positive excess everywhere.
            "claim": alpha,
            "omit": set(),
        }
        certificate = Certificate(alpha, brackets[index])
        # Component modes depend on the z-bin envelopes, not on alpha.
        certificate.mode_cache = shared_mode_cache
        with contextlib.redirect_stdout(io.StringIO()):
            result = certificate.run()
        total = arb(result["computed_total_lower_ball"])
        bound_num = certified_decimal_numerator(total, places)
        alpha_num = index * 10 ** (places - 3)
        excess_num = bound_num - alpha_num
        if excess_num <= 0:
            raise RuntimeError(f"nonpositive certified decimal excess at {alpha}")
        max_b_num = max(row["b_num"] for row in brackets[index]["rows"])
        row = {
            "alpha": alpha,
            "certified_fdr_lower_bound": fixed_decimal(bound_num, places),
            "certified_excess_lower_bound": fixed_decimal(excess_num, places),
            "computed_total_lower_ball": result["computed_total_lower_ball"],
            "initial_cutoff": f"{cstart_num / 1000:.3f}",
            "maximum_feasible_witness": f"{max_b_num / WITNESS_DENOMINATOR:.8f}",
            "maximum_prefix_depth": result["maximum_prefix_depth"],
        }
        rows.append(row)
        print(
            alpha,
            row["certified_fdr_lower_bound"],
            row["certified_excess_lower_bound"],
            "depth",
            row["maximum_prefix_depth"],
            flush=True,
        )
    if len(shared_mode_cache) != 3000:
        raise RuntimeError(
            f"expected 3000 distinct component modes, got {len(shared_mode_cache)}"
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--decimal-places", type=int, default=12)
    args = parser.parse_args()
    if args.decimal_places < 3:
        parser.error("--decimal-places must be at least 3")

    base = Path(__file__).resolve().parent
    default_output = base.parents[1] / "reproduced" / "central_three_block"
    output_csv = args.output_csv or default_output / "certified_lower_bound_curve.csv"
    output_json = args.output_json or default_output / "certified_lower_bound_curve.json"

    brackets = generate_candidate_brackets()
    check_archived_brackets(base, brackets)
    rows = certify_grid(brackets, args.decimal_places)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "alpha",
                "certified_fdr_lower_bound",
                "certified_excess_lower_bound",
            ],
        )
        writer.writeheader()
        writer.writerows(
            {key: row[key] for key in writer.fieldnames} for row in rows
        )

    payload = {
        "description": "Certified lower bound from SI Eq. (certified-sum)",
        "alpha_grid": "0.001,0.002,...,0.100",
        "decimal_values": (
            "Exact decimals rounded downward and rechecked against each final "
            "outward-rounded Arb ball"
        ),
        "arithmetic": "python-flint Arb, 100 decimal digits, outward rounding",
        "factor_bins": {"minimum": -6, "maximum": 6, "mesh": "1/100"},
        "terminal_prefix_width": "1/5000",
        "candidate_witness_rule": (
            "b=(floor(10^8*c_root)+501)/10^8; every b is independently "
            "verified by Arb"
        ),
        "rows": rows,
    }
    output_json.write_text(json.dumps(payload, indent=2) + "\n")
    print("wrote", output_csv, "and", output_json)


if __name__ == "__main__":
    main()
