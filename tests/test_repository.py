from __future__ import annotations

from fractions import Fraction
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
CENTRAL = ROOT / "experiments" / "central_three_block"
SIX = ROOT / "experiments" / "larger_violation_six_block"
FINITE85 = ROOT / "experiments" / "finite_85_tests"
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(CENTRAL))

from verify_results import validate_scientific_results  # noqa: E402


class ModelInvariantTests(unittest.TestCase):
    def test_central_model_has_unit_variances(self) -> None:
        loadings = [Fraction(3, 10), Fraction(2, 11), Fraction(20, 21)]
        residual_variances = [Fraction(91, 100), Fraction(117, 121), Fraction(41, 441)]
        variances = [
            rho * rho + variance
            for rho, variance in zip(loadings, residual_variances)
        ]
        self.assertEqual(variances, [1, 1, 1])
        self.assertTrue(all(rho > 0 for rho in loadings))
        self.assertTrue(all(variance > 0 for variance in residual_variances))
        self.assertEqual(sum([163, 1, 3]), 167)

    def test_archived_manuscript_values(self) -> None:
        validate_scientific_results()

    def test_tiny_count_thinning_smoke_run(self) -> None:
        from central_three_block_finite_sample_mc import run_task

        result = run_task((1, 0, 0.05, 200, 100, 12345))
        self.assertEqual(result["N"], 1)
        self.assertEqual(result["m"], 167)
        self.assertEqual(result["replications"], 200)
        self.assertGreaterEqual(result["fdr"], 0.0)
        self.assertLessEqual(result["fdr"], 1.0)
        self.assertGreaterEqual(result["maximum_fixed_point_iterations"], 1)


class CommandLineTests(unittest.TestCase):
    def test_finite_85_refuses_an_unmanaged_chain_directory(self) -> None:
        script = FINITE85 / "bh_finite_multiblock_m85_certificate.py"
        with tempfile.TemporaryDirectory(prefix="bh-m85-safety-test-") as temporary:
            workdir = Path(temporary)
            sentinel = workdir / "keep-me.txt"
            sentinel.write_text("sentinel\n", encoding="utf-8")
            completed = subprocess.run(
                [sys.executable, str(script), "--chain", "0", str(workdir)],
                cwd=ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=30,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("refusing unmanaged", completed.stdout)
            self.assertTrue(sentinel.is_file())

    def test_public_scripts_have_working_help(self) -> None:
        scripts = [
            CENTRAL / "central_three_block_certificate.py",
            CENTRAL / "central_three_block_certificate_grid.py",
            CENTRAL / "central_three_block_finite_sample_mc.py",
            CENTRAL / "central_three_block_limiting_curve.py",
            CENTRAL / "central_three_block_selected_levels_mc.py",
            CENTRAL / "central_three_block_tail_stratified_mc.py",
            CENTRAL / "plot_realized_fdr_vs_alpha.py",
            FINITE85 / "bh_finite_multiblock_m85_certificate.py",
            SIX / "larger_violation_certificate.py",
            SIX / "larger_violation_numerical_evaluation.py",
            SIX / "larger_violation_stratified_mc.py",
            ROOT / "scripts" / "checksums.py",
            ROOT / "scripts" / "reproduce.py",
            ROOT / "scripts" / "verify_results.py",
        ]
        for script in scripts:
            with self.subTest(script=script.name):
                completed = subprocess.run(
                    [sys.executable, str(script), "--help"],
                    cwd=ROOT,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=30,
                )
                self.assertEqual(completed.returncode, 0, completed.stdout)
                self.assertIn("usage", completed.stdout.lower())


if __name__ == "__main__":
    unittest.main()
