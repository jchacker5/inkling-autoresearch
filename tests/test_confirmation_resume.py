import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("campaign", Path(__file__).parents[1] / "swarm" / "campaign.py")
campaign = importlib.util.module_from_spec(spec)
stubs = {
    "prepare": types.SimpleNamespace(HOLDOUT_EVAL_SEED=99991),
    "confirm": types.SimpleNamespace(BASELINE_LANE="baseline-holdout", CONFIRM_SEEDS=range(100, 106),
                                    best_candidate=None, confirm_table=None, write_tsv=None),
}
with patch.dict(sys.modules, stubs):
    spec.loader.exec_module(campaign)


class ConfirmationResume(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.wt = Path(self.directory.name)
        self.rows = []
        self.marked = []
        self.argv = ("confirm-recipe-abc-100", "recipe-confirm", 100, self.wt, "abc", "h", "b")
        self.good = dict(lane="recipe-confirm", commit="abcdef0", seed="100", harness="h",
                         base="b", eval_seed="99991", valid="True", verdict="holdout", dirty="False")
        self.checkout_failed = False

    def execute(self, row=None, checkout=True):
        def git(args, **kw):
            if self.checkout_failed and "checkout" in args:
                raise subprocess.CalledProcessError(1, args)
            return types.SimpleNamespace(stdout="abcdef0\n" if "rev-parse" in args else "")
        def runner(*args, **kw):
            if row:
                self.rows.append(row)
            return 0
        with patch.object(campaign, "ledger", side_effect=lambda: list(self.rows)), \
                patch.object(campaign, "STATE", self.wt), patch.object(campaign, "mark", self.marked.append), \
                patch.object(campaign, "wait_for_power"), patch.object(campaign, "run", side_effect=runner) as run, \
                patch.object(campaign.subprocess, "run", side_effect=git):
            try:
                campaign.confirm_seed(*self.argv, desc="test fixture", checkout=checkout)
            finally:
                self.training_calls = run.call_count

    def test_failed_checkout_never_trains_or_marks(self):
        self.checkout_failed = True
        with self.assertRaises(subprocess.CalledProcessError):
            self.execute(self.good)
        self.assertEqual(self.training_calls, 0)
        self.assertEqual(self.marked, [])

    def test_failed_zero_exit_retries_despite_stale_marker(self):
        marker = self.wt / (self.argv[0] + ".done")
        marker.touch()
        with self.assertRaisesRegex(RuntimeError, "valid receipt"):
            self.execute({**self.good, "valid": "False"})
        self.assertEqual(self.marked, [])
        self.assertFalse(marker.exists())
        self.execute(self.good)
        self.assertEqual(self.marked, [self.argv[0]])
        self.assertEqual(self.training_calls, 1)

    def test_valid_receipt_resume_never_retrains(self):
        self.rows.append(self.good)
        self.execute()
        self.assertEqual(self.training_calls, 0)

    def test_wrong_harness_does_not_complete(self):
        with self.assertRaisesRegex(RuntimeError, "valid receipt"):
            self.execute({**self.good, "harness": "other"})
        self.assertEqual(self.marked, [])

    def test_duplicate_receipts_stop_instead_of_training(self):
        self.rows.extend([self.good, self.good])
        with self.assertRaisesRegex(RuntimeError, "duplicate"):
            self.execute()
        self.assertEqual(self.training_calls, 0)

    def test_baseline_revision_cannot_drift_on_resume(self):
        with patch.object(campaign, "STATE", self.wt), patch.object(campaign.subprocess, "run") as git:
            git.return_value.stdout = "original\n"
            self.assertEqual(campaign.pinned_confirmation_baseline("h", "b"), "original")
            git.return_value.stdout = "advanced\n"
            self.assertEqual(campaign.pinned_confirmation_baseline("h", "b"), "original")
            self.assertEqual(git.call_count, 1)
