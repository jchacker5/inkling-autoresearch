import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("confirm", Path(__file__).parents[1] / "swarm" / "confirm.py")
confirm = importlib.util.module_from_spec(spec)
with patch.dict(sys.modules, {"run": types.SimpleNamespace(BPB_GATE=.02, PRIMARIES={"val_bpb": -1})}):
    spec.loader.exec_module(confirm)


def make_rows():
    candidate = dict(lane="recipe", commit="candidate", valid="True", verdict="candidate",
                     primary="val_bpb", t_primary="5", harness="same", base="v4")
    def draw(lane, seed, value, commit):
        return dict(lane=lane, seed=str(seed), commit=commit, val_bpb=str(value), valid="True",
                    eval_seed="99991", harness="same", base="v4")
    base = [draw(confirm.BASELINE_LANE, s, 1.5, "baseline") for s in confirm.CONFIRM_SEEDS]
    treatment = [draw("recipe-confirm", s, 1.4, "candidate") for s in confirm.CONFIRM_SEEDS]
    return candidate, base, treatment


class ConfirmationIntegrity(unittest.TestCase):
    def score(self, candidate, base, treatment):
        return confirm.confirm_table([candidate, *base, *treatment], ["recipe"], 99991)[0]

    def test_repeated_seed_is_not_six_replicates(self):
        c, b, t = make_rows()
        result = self.score(c, b[:1] * 6, t[:1] * 6)
        self.assertFalse(result["promoted"])
        self.assertEqual(result["n_cand"], 1)

    def test_incomplete_seed_matrix_cannot_promote(self):
        c, b, t = make_rows()
        self.assertFalse(self.score(c, b, t[:-1])["promoted"])

    def test_mixed_harness_cannot_promote(self):
        c, b, t = make_rows()
        t[0]["harness"] = "different"
        self.assertFalse(self.score(c, b, t)["promoted"])

    def test_complete_clean_seed_matrix_is_scored(self):
        c, b, t = make_rows()
        self.assertTrue(self.score(c, b, t)["promoted"])
