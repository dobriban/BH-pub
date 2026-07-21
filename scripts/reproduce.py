#!/usr/bin/env python3
"""Cross-platform entry point for checks and manuscript computations."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
CENTRAL = ROOT / "experiments" / "central_three_block"
FINITE85 = ROOT / "experiments" / "finite_85_tests"
SIX = ROOT / "experiments" / "larger_violation_six_block"
REFERENCE = ROOT / "results" / "reference"
OUTPUT = ROOT / "reproduced"
PYTHON = sys.executable


def display_command(command: list[object]) -> str:
    return " ".join(
        f'"{item}"' if " " in str(item) else str(item) for item in command
    )


def run(command: list[object], transcript: Path | None = None) -> None:
    args = [str(item) for item in command]
    print(f"+ {display_command(args)}", flush=True)
    environment = os.environ.copy()
    environment.setdefault("MPLBACKEND", "Agg")
    if transcript is None:
        subprocess.run(args, cwd=ROOT, env=environment, check=True)
        return
    transcript.parent.mkdir(parents=True, exist_ok=True)
    with transcript.open("w", encoding="utf-8", newline="\n") as handle:
        process = subprocess.Popen(
            args,
            cwd=ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            handle.write(line)
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, args)


def check() -> None:
    run([PYTHON, ROOT / "scripts" / "verify_results.py"])
    run(
        [
            PYTHON,
            "-m",
            "compileall",
            "-q",
            ROOT / "experiments",
            ROOT / "scripts",
            ROOT / "tests",
        ]
    )
    run([PYTHON, "-m", "unittest", "discover", "-s", ROOT / "tests", "-v"])


def figure() -> None:
    destination = OUTPUT / "figures"
    destination.mkdir(parents=True, exist_ok=True)
    central_reference = REFERENCE / "central_three_block"
    run(
        [
            PYTHON,
            CENTRAL / "plot_realized_fdr_vs_alpha.py",
            "--curve",
            central_reference / "tail_stratified_mc_curve.csv",
            "--certified",
            central_reference / "certified_lower_bound_curve.csv",
            "--finite",
            central_reference / "finite_sample_fdr_curve.csv",
            "--output-prefix",
            destination / "realized_fdr_vs_alpha",
            "--violation-output",
            destination / "realized_fdr_violation.pdf",
        ]
    )


def theorem_certificates() -> None:
    destination = OUTPUT / "central_three_block"
    destination.mkdir(parents=True, exist_ok=True)
    for alpha in ("0.01", "0.05", "0.10"):
        run(
            [
                PYTHON,
                CENTRAL / "central_three_block_certificate.py",
                "--alpha",
                alpha,
                "--output",
                destination / f"certificate_alpha_{alpha}.json",
            ],
            destination / f"certificate_alpha_{alpha}.txt",
        )


def certificate_grid() -> None:
    destination = OUTPUT / "central_three_block"
    destination.mkdir(parents=True, exist_ok=True)
    run(
        [
            PYTHON,
            CENTRAL / "central_three_block_certificate_grid.py",
            "--output-csv",
            destination / "certified_lower_bound_curve.csv",
            "--output-json",
            destination / "certified_lower_bound_curve.json",
        ]
    )


def limiting_curve() -> None:
    destination = OUTPUT / "central_three_block"
    destination.mkdir(parents=True, exist_ok=True)
    run(
        [
            PYTHON,
            CENTRAL / "central_three_block_limiting_curve.py",
            "--output-dir",
            destination,
        ]
    )


def limiting_mc() -> None:
    limiting_curve()
    destination = OUTPUT / "central_three_block"
    run(
        [
            PYTHON,
            CENTRAL / "central_three_block_tail_stratified_mc.py",
            "--output-dir",
            destination,
            "--quadrature",
            destination / "limiting_fdr_curve.csv",
        ]
    )


def finite_mc() -> None:
    destination = OUTPUT / "central_three_block"
    destination.mkdir(parents=True, exist_ok=True)
    run(
        [
            PYTHON,
            CENTRAL / "central_three_block_finite_sample_mc.py",
            "--output-dir",
            destination,
        ],
        destination / "finite_sample_mc_run.txt",
    )


def selected_mc() -> None:
    destination = OUTPUT / "central_three_block"
    destination.mkdir(parents=True, exist_ok=True)
    run(
        [
            PYTHON,
            CENTRAL / "central_three_block_selected_levels_mc.py",
            "--output-dir",
            destination,
        ]
    )


def finite_85() -> None:
    destination = OUTPUT / "finite_85_tests"
    destination.mkdir(parents=True, exist_ok=True)
    run(
        [PYTHON, FINITE85 / "bh_finite_multiblock_m85_certificate.py"],
        destination / "bh_finite_multiblock_m85_certificate_output.txt",
    )


def six_numerical() -> None:
    destination = OUTPUT / "larger_violation_six_block"
    destination.mkdir(parents=True, exist_ok=True)
    run(
        [
            PYTHON,
            SIX / "larger_violation_numerical_evaluation.py",
            "--output",
            destination / "larger_violation_numerical_results.json",
        ],
        destination / "larger_violation_numerical_output.txt",
    )


def six_mc() -> None:
    destination = OUTPUT / "larger_violation_six_block"
    destination.mkdir(parents=True, exist_ok=True)
    run(
        [
            PYTHON,
            SIX / "larger_violation_stratified_mc.py",
            "--seed",
            "20261001",
            "--replications",
            "30",
            "--output-prefix",
            destination / "larger_violation_mc",
        ],
        destination / "larger_violation_mc_summary.txt",
    )


def six_certificate() -> None:
    destination = OUTPUT / "larger_violation_six_block"
    destination.mkdir(parents=True, exist_ok=True)
    run(
        [
            PYTHON,
            SIX / "larger_violation_certificate.py",
            "--alpha",
            "0.05",
            "--output",
            destination / "certificate_alpha_0.05.json",
        ],
        destination / "certificate_alpha_0.05.txt",
    )


COMMANDS = {
    "check": check,
    "figure": figure,
    "theorem-certificates": theorem_certificates,
    "certificate-grid": certificate_grid,
    "limiting-curve": limiting_curve,
    "limiting-mc": limiting_mc,
    "finite-mc": finite_mc,
    "selected-mc": selected_mc,
    "finite-85": finite_85,
    "six-numerical": six_numerical,
    "six-mc": six_mc,
    "six-certificate": six_certificate,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=COMMANDS)
    args = parser.parse_args()
    COMMANDS[args.command]()


if __name__ == "__main__":
    main()
