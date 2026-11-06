"""Train and evaluate applicability and observed-benefit gates without LLM updates."""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import sklearn

from latentskill.models.boundary_gate import BoundaryGate


def validate_rows(rows):
    groups = defaultdict(list)
    family_splits = defaultdict(set)
    ids = set()
    for row in rows:
        if row["applicable"] not in (0, 1):
            raise ValueError("Applicability labels must be binary")
        if row["id"] in ids:
            raise ValueError(f"Duplicate state: {row['id']}")
        ids.add(row["id"])
        if row["split"] not in {"train", "val", "test"}:
            raise ValueError("Unknown split")
        groups[row["pair_id"]].append(row)
        family_splits[Path(row["gamefile"]).parent.parent.name].add(row["split"])
    if any(len(splits) != 1 for splits in family_splits.values()):
        raise ValueError("Task family leaked across splits")
    for pair in groups.values():
        if (len(pair) != 2 or {r["applicable"] for r in pair} != {0, 1}
                or len({r["split"] for r in pair}) != 1
                or len({r["skill"] for r in pair}) != 1
                or len({r["gamefile"] for r in pair}) != 1):
            raise ValueError("Each pair needs two complementary states in one split")
    if {r["split"] for r in rows} != {"train", "val", "test"}:
        raise ValueError("Train, validation, and test splits are required")


def mean(values):
    return float(np.mean(values)) if len(values) else None


def metrics(rows, activated):
    truth = np.array([r["applicable"] for r in rows], dtype=bool)
    selected = np.array(activated, dtype=bool)
    result = {
        "states": len(rows), "pairs": len({r["pair_id"] for r in rows}),
        "activation_rate": mean(selected), "boundary_accuracy": mean(selected == truth),
        "false_activation_rate": mean(selected[~truth]),
        "missed_activation_rate": mean(~selected[truth]),
    }
    if all("on" in row and "off" in row for row in rows):
        on = np.array([r["on"]["reward"] for r in rows])
        off = np.array([r["off"]["reward"] for r in rows])
        reward = np.where(selected, on, off)
        result.update({
            "next_action_accuracy": mean(reward),
            "gain_vs_off": mean(reward - off),
            "negative_transfer_vs_off": mean(np.maximum(off - reward, 0)),
            "harmful_activation_rate": mean(selected & (on < off)),
            "applicable_accuracy": mean(reward[truth]),
            "inapplicable_accuracy": mean(reward[~truth]),
            "beneficial_states": int(np.sum(on > off)),
            "harmful_states": int(np.sum(on < off)),
            "tied_states": int(np.sum(on == off)),
        })
    return result


def select_threshold(gate, val, target):
    probabilities = [gate.probability(r) for r in val]
    # Include never-on and always-on, without touching the test split.
    thresholds = sorted({0.0, 0.5, 1.000001, *probabilities})
    def objective(threshold):
        selected = np.array(probabilities) >= threshold
        if target == "benefit":
            value = mean([r["on" if active else "off"]["reward"]
                          for r, active in zip(val, selected)])
        else:
            value = metrics(val, selected)["boundary_accuracy"]
        # Prefer fewer activations on tied validation scores.
        return value, -float(np.mean(selected)), threshold
    return max(thresholds, key=objective)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Pairs or LoRA outcome JSONL")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rows = [json.loads(line) for line in Path(args.input).read_text().splitlines() if line]
    validate_rows(rows)
    splits = {split: [r for r in rows if r["split"] == split]
              for split in ("train", "val", "test")}
    train, val, test = (splits[s] for s in ("train", "val", "test"))
    for split in splits:
        if {r["skill"] for r in splits[split]} != {r["skill"] for r in train}:
            raise ValueError("Every split must cover all skills")
    outcomes = all("on" in r and "off" in r for r in rows)
    if not outcomes and any("on" in r or "off" in r for r in rows):
        raise ValueError("Outcome records must have complete on/off results for every state")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "seed": args.seed, "input": str(Path(args.input).resolve()),
        "input_sha256": hashlib.sha256(Path(args.input).read_bytes()).hexdigest(),
        "versions": {"numpy": np.__version__, "scikit_learn": sklearn.__version__},
        "split_sizes": {key: len(value) for key, value in splits.items()},
        "scope": "real ALFWorld next-action probe; not full-episode performance",
        "has_frozen_lora_outcomes": outcomes, "results": {},
    }
    if outcomes:
        report["outcome_gain_counts"] = {
            split: {
                "beneficial": sum(r["on"]["reward"] > r["off"]["reward"] for r in data),
                "harmful": sum(r["on"]["reward"] < r["off"]["reward"] for r in data),
                "tied": sum(r["on"]["reward"] == r["off"]["reward"] for r in data),
            }
            for split, data in splits.items()
        }
    policies = {"always_off": [False] * len(test), "always_on": [True] * len(test)}
    gates = {}
    for target in (["applicability", "benefit"] if outcomes else ["applicability"]):
        labels = [int(r["on"]["reward"] > r["off"]["reward"])
                  if target == "benefit" else r["applicable"] for r in train]
        gate = BoundaryGate(seed=args.seed).fit(train, labels)
        gate.threshold = select_threshold(gate, val, target)
        gate.save(output / f"{target}_gate.pkl")
        gates[target] = gate
        policies[target] = [gate.activate(row) for row in test]
        report[f"{target}_training"] = {
            "positive_labels": sum(labels), "negative_labels": len(labels) - sum(labels),
            "validation_threshold": gate.threshold,
            "constant_heads": {s: h for s, h in gate.heads.items() if isinstance(h, float)},
        }
    if outcomes:
        policies["oracle"] = [r["on"]["reward"] > r["off"]["reward"] for r in test]
    for name, selected in policies.items():
        report["results"][name] = metrics(test, selected)
        report["results"][name]["per_skill"] = {
            skill: metrics([r for r in test if r["skill"] == skill],
                           [a for r, a in zip(test, selected) if r["skill"] == skill])
            for skill in sorted({r["skill"] for r in test})
        }
    with (output / "test_predictions.jsonl").open("w") as handle:
        for row in test:
            handle.write(json.dumps({
                "id": row["id"], "pair_id": row["pair_id"], "skill": row["skill"],
                "applicable": row["applicable"],
                "gates": {name: {"probability": gate.probability(row),
                                 "activated": gate.activate(row)}
                          for name, gate in gates.items()},
            }) + "\n")
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
