import copy
import unittest

from evals.boundary_gate.stage2a_collect import parse_action, policy_prompt
from evals.boundary_gate.stage2a_data import validate_splits
from evals.boundary_gate.stage2a_train import features, metrics, validate_outcomes


class Stage2ATests(unittest.TestCase):
    def test_scene_leakage_rejected_even_for_different_games(self):
        rows = [{"gamefile": "a", "family": "a", "scene": "1", "split": "train"},
                {"gamefile": "b", "family": "b", "scene": "1", "split": "validation"}]
        with self.assertRaisesRegex(ValueError, "scene"):
            validate_splits(rows)

    def test_prompt_retains_absolute_step_and_last_five_observations(self):
        history = [(f"obs{i}", f"go to cabinet {i}") for i in range(8)]
        prompt = policy_prompt("goal", "current", ["look"], history)
        self.assertIn("8 step(s)", prompt)
        self.assertIn("step 9", prompt)
        self.assertNotIn("obs2", prompt)
        self.assertIn("obs3", prompt)

    def test_native_parser_requires_tags_and_preserves_invalid_commands(self):
        self.assertEqual(parse_action("go to fridge 1", ["go to fridge 1"]), ("look", False, False))
        self.assertEqual(parse_action("<action>GO  TO fridge 1</action>", ["go to fridge 1"]),
                         ("go to fridge 1", True, True))
        self.assertEqual(parse_action("<action>fly away</action>", ["look"]), ("fly away", True, False))

    def test_gate_features_do_not_use_labels_replays_or_candidate_origin(self):
        row = dict(task_type="cool", goal="goal", observation="obs", inventory="empty",
                   history=["look"], admissible_actions=["look"], gain=1,
                   scene="123", family="secret", proposal="positive", on={"reward": 1})
        changed = dict(row, gain=-1, scene="999", family="other", proposal="negative",
                       on={"reward": 0}, replay=["secret future"])
        self.assertEqual(features(row), features(changed))

    def test_return_labels_and_matched_state_are_enforced(self):
        game = dict(gamefile="a", family="f", scene="1", split="train", task_type="cool")
        branch = dict(reward=0, start={"observation": "same"}, remaining_budget=50,
                      steps_log=[{"input_ids_sha256": "same"}])
        row = dict(game, id="id", gain=0, replay=[], snapshot=branch["start"],
                   on=copy.deepcopy(branch), off=copy.deepcopy(branch))
        validate_outcomes([row], {"records": [game]})
        row["gain"] = 1
        with self.assertRaisesRegex(ValueError, "return difference"):
            validate_outcomes([row], {"records": [game]})
        row["gain"] = 0
        row["on"]["start"] = {"observation": "different"}
        with self.assertRaisesRegex(ValueError, "same replayed state"):
            validate_outcomes([row], {"records": [game]})

    def test_metrics_preserve_positive_and_avoid_negative_returns(self):
        rows = [dict(family=str(i), gamefile=str(i), off={"reward": off}, on={"reward": on})
                for i, (off, on) in enumerate([(0, 1), (1, 0), (0, 0)])]
        result = metrics(rows, [True, False, False])
        self.assertEqual(result["positive_transfer_retention"], 1.)
        self.assertEqual(result["negative_transfer_rate"], 0.)
        self.assertEqual(result["selected_remaining_success_rate"], 2 / 3)


if __name__ == "__main__":
    unittest.main()
