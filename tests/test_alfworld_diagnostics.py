"""Checks for paired task selection and episode-return comparisons."""

import json
from pathlib import Path
import tempfile
import unittest

from evals.boundary_gate.diagnose import paired_summary, restrict_games
from evals.boundary_gate.batched_diagnose import paired_batches


class FullTaskDiagnosticsTests(unittest.TestCase):
    def test_batching_never_splits_base_on_counterfactuals(self):
        requests = {key: key for key in (
            "00:base", "00:on", "00:text", "01:base", "02:base", "02:on", "02:text",
        )}
        batches = list(paired_batches(requests, 4))
        self.assertEqual(sorted(x for batch in batches for x in batch), sorted(requests))
        for game in ("00", "02"):
            self.assertTrue(any(f"{game}:base" in batch and f"{game}:on" in batch for batch in batches))
        self.assertTrue(all(len(batch) <= 4 for batch in batches))
        with self.assertRaises(ValueError):
            list(paired_batches(requests, 1))

    def test_manifest_rejects_missing_and_duplicate_games(self):
        with tempfile.TemporaryDirectory() as folder:
            manifest = Path(folder) / "games.json"
            manifest.write_text(json.dumps(["/a", "/b"]))
            self.assertEqual(restrict_games(["/b", "/a", "/c"], manifest), ["/a", "/b"])
            with self.assertRaises(ValueError):
                restrict_games(["/a"], manifest)
            manifest.write_text(json.dumps(["/a", "/a"]))
            with self.assertRaises(ValueError):
                restrict_games(["/a"], manifest)

    def test_episode_oracle_and_transfer_use_final_success(self):
        def row(won):
            return {"won": won, "steps": 1,
                    "steps_log": [{"in_admissible": True, "hit_token_limit": False}]}
        result = paired_summary({
            "base": {"a": row(True), "b": row(False), "c": row(False)},
            "on": {"a": row(False), "b": row(True), "c": row(False)},
            "text": {"a": row(False), "b": row(False), "c": row(True)},
        })
        self.assertEqual(result["on"]["beneficial_games"], 1)
        self.assertEqual(result["on"]["harmful_games"], 1)
        self.assertEqual(result["on"]["neutral_games"], 1)
        self.assertEqual(result["oracle_base_on"]["successes"], 2)
        self.assertEqual(result["oracle_all_three"]["successes"], 3)
        with self.assertRaises(ValueError):
            paired_summary({"base": {"a": row(True)}, "on": {}, "text": {}})


if __name__ == "__main__":
    unittest.main()
