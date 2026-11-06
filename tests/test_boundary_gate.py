"""Run with: python -m unittest discover -s tests -v."""

import copy
import tempfile
import unittest
from pathlib import Path

from evals.boundary_gate.train import metrics, select_threshold, validate_rows
from latentskill.models.boundary_gate import BoundaryGate, state_text


def sample_rows():
    rows = []
    for split in ("train", "val", "test"):
        for applicable in (0, 1):
            rows.append({
                "id": f"{split}:{applicable}", "pair_id": split, "split": split,
                "skill": "heat", "gamefile": f"/{split}/family-{split}/trial/game.tw-pddl",
                "goal": "heat an apple", "inventory": "carrying apple 1",
                "observation": "microwave 1" if applicable else "cabinet 1",
                "history": [], "applicable": applicable,
                "on": {"reward": applicable}, "off": {"reward": 1 - applicable},
            })
    return rows


class BoundaryGateTests(unittest.TestCase):
    def test_rejects_split_pair_and_family_leakage(self):
        rows = sample_rows()
        validate_rows(rows)
        broken = copy.deepcopy(rows)
        broken[0]["split"] = "test"
        with self.assertRaises(ValueError):
            validate_rows(broken)
        broken = copy.deepcopy(rows)
        broken[-1]["gamefile"] = broken[0]["gamefile"]
        with self.assertRaises(ValueError):
            validate_rows(broken)

    def test_oracle_metadata_cannot_enter_features(self):
        row = sample_rows()[0]
        text = state_text(row)
        for key in ("applicable", "expected_action", "pair_id", "id", "split", "on", "off"):
            row[key] = "LEAKED_LABEL"
        self.assertEqual(text, state_text(row))

    def test_metrics_count_negative_transfer_and_conditional_errors(self):
        rows = sample_rows()[:2]
        always_on = metrics(rows, [True, True])
        self.assertEqual(always_on["false_activation_rate"], 1)
        self.assertEqual(always_on["missed_activation_rate"], 0)
        self.assertEqual(always_on["negative_transfer_vs_off"], 0.5)
        self.assertEqual(always_on["gain_vs_off"], 0)
        gated = metrics(rows, [False, True])
        self.assertEqual(gated["next_action_accuracy"], 1)
        self.assertEqual(gated["negative_transfer_vs_off"], 0)

    def test_tied_outcomes_fall_back_without_enabling_adapter(self):
        rows = sample_rows()
        for row in rows:
            row["on"]["reward"] = row["off"]["reward"] = 1
        gate = BoundaryGate().fit(rows[:2], [0, 0])
        gate.threshold = select_threshold(gate, rows[2:4], "benefit")
        adapter = {"fixed": object()}
        self.assertIsNone(gate.select_adapter(rows[-1], adapter))
        gate.threshold = 0
        self.assertIs(gate.select_adapter(rows[-1], adapter), adapter)
        unseen = dict(rows[-1], skill="unknown")
        self.assertIsNone(gate.select_adapter(unseen, adapter))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gate.pkl"
            gate.save(path)
            self.assertEqual(BoundaryGate.load(path).probability(rows[-1]), 0)


if __name__ == "__main__":
    unittest.main()
