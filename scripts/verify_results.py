#!/usr/bin/env python3
"""Validate the archived values used in the PNAS draft."""

from __future__ import annotations

import argparse
import csv
from decimal import Decimal
import json
import math
from pathlib import Path
import re

from checksums import verify_manifest


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "results" / "reference"
CENTRAL = REFERENCE / "central_three_block"
FINITE85 = REFERENCE / "finite_85_tests"
SIX = REFERENCE / "larger_violation_six_block"


def load_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def ball_center(text: str) -> Decimal:
    match = re.match(r"^\[([^ ]+) \+/-", text)
    if match is None:
        raise AssertionError(f"not an Arb ball: {text!r}")
    return Decimal(match.group(1))


def assert_close(actual: float, expected: float, tolerance: float = 5e-12) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance):
        raise AssertionError(f"{actual!r} != {expected!r} within {tolerance}")


def validate_central_certificates() -> None:
    expected = {
        "0.01": Decimal("0.011196841965752766090603159872269028278867491834317585"),
        "0.05": Decimal("0.054852568515632064879803733029907669658887518742603379"),
        "0.10": Decimal("0.105337158461421569675237713019096646043824513224196007"),
    }
    advertised = {
        "0.01": Decimal("0.0111"),
        "0.05": Decimal("0.0548"),
        "0.10": Decimal("0.1053"),
    }
    for alpha, center in expected.items():
        payload = load_json(CENTRAL / f"certificate_alpha_{alpha}.json")
        assert payload["alpha"] == alpha
        assert payload["omitted_z_bins"] == []
        assert payload["certified_factor_range"] == ["-6", "6"]
        assert payload["z_mesh"] == "1/100"
        assert len(payload["bins"]) == 1200
        actual = ball_center(payload["computed_total_lower_ball"])
        assert abs(actual - center) < Decimal("1e-54")
        assert actual > advertised[alpha] > Decimal(alpha)


def validate_full_grid() -> None:
    with (CENTRAL / "certified_lower_bound_curve.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 100
    alphas = [Decimal(row["alpha"]) for row in rows]
    excesses = [Decimal(row["certified_excess_lower_bound"]) for row in rows]
    assert alphas == [Decimal(index) / 1000 for index in range(1, 101)]
    assert all(value > 0 for value in excesses)
    assert min(zip(excesses, alphas)) == (Decimal("0.000014949530"), Decimal("0.001"))
    assert max(zip(excesses, alphas)) == (Decimal("0.005603288950"), Decimal("0.083"))


def validate_finite_simulations() -> None:
    full = load_json(CENTRAL / "finite_sample_fdr_curve.json")
    assert full["Ns"] == [50, 100, 500, 1000]
    assert full["replications_per_N_alpha"] == 100000
    assert full["base_seed"] == 20260821
    assert len(full["rows"]) == 400
    above_counts = {
        n_value: sum(
            row["ci95_lower"] > row["alpha"]
            for row in full["rows"]
            if row["N"] == n_value
        )
        for n_value in full["Ns"]
    }
    assert above_counts == {50: 87, 100: 89, 500: 92, 1000: 95}

    selected = load_json(CENTRAL / "finite_sample_selected_levels.json")
    assert selected["replications_per_N_alpha"] == 500000
    assert selected["base_seed"] == 20260921
    assert len(selected["rows"]) == 12
    expected = {
        (1000, 0.01): 0.010610874635897521,
        (1000, 0.05): 0.05313367299745379,
        (1000, 0.10): 0.10323011071322076,
    }
    indexed = {(row["N"], row["alpha"]): row for row in selected["rows"]}
    for key, value in expected.items():
        assert_close(indexed[key]["fdr"], value, 1e-16)
        assert indexed[key]["ci95_lower"] > key[1]


def validate_finite_85() -> None:
    text = (FINITE85 / "bh_finite_multiblock_m85_certificate_output.txt").read_text(
        encoding="utf-8"
    )
    assert "0.0001000092703599461" in text
    assert "CERTIFIED: FDR > 0.000100005 > alpha = 0.0001" in text


def validate_six_block() -> None:
    parameters = load_json(
        ROOT / "experiments" / "larger_violation_six_block" / "model_parameters.json"
    )
    weights = [Decimal(value) for value in parameters["weights"]]
    means = [Decimal(value) for value in parameters["means"]]
    loadings = [Decimal(value) for value in parameters["loadings"]]
    denominator = Decimal(parameters["common_block_denominator"])
    assert sum(weights) == Decimal(1)
    assert all((weight * denominator) == (weight * denominator).to_integral() for weight in weights)
    assert means[0] == 0
    assert all(abs(value) < 1 for value in loadings)

    numerical = load_json(SIX / "larger_violation_numerical_results.json")
    numerical_expected = {
        "0.01": 0.011424561004499635,
        "0.05": 0.06204595215960603,
        "0.10": 0.11760402520651653,
    }
    for alpha, value in numerical_expected.items():
        assert_close(numerical["results"][alpha]["limiting_fdr"], value, 1e-15)

    monte_carlo = load_json(SIX / "larger_violation_mc.json")["summary"]
    mc_expected = {
        "0.01": 0.011428727525522915,
        "0.05": 0.0620593705420321,
        "0.10": 0.11763103991484726,
    }
    for alpha, value in mc_expected.items():
        assert_close(monte_carlo["alphas"][alpha]["estimate"], value, 1e-15)

    certificate = load_json(SIX / "certificate_alpha_0.05.json")
    assert len(certificate["results"]) == 1
    result = certificate["results"][0]
    assert result["alpha"] == "0.05"
    assert len(result["bins"]) == 160
    assert ball_center(result["computed_total_lower_ball"]) > Decimal("0.0582252019310")


def validate_scientific_results() -> None:
    validate_central_certificates()
    validate_full_grid()
    validate_finite_simulations()
    validate_finite_85()
    validate_six_block()
    print("validated all archived manuscript-level results")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-checksums",
        action="store_true",
        help="validate values without checking SHA256SUMS",
    )
    args = parser.parse_args()
    if not args.skip_checksums:
        verify_manifest()
    validate_scientific_results()


if __name__ == "__main__":
    main()
