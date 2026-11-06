"""Compare gates on scene-isolated real remaining-return counterfactual labels."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import pickle

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

from evals.boundary_gate.stage2a_data import digest, validate_splits


def features(row):
    """Whitelist only information available before choosing On/Off."""
    return (f"Task type: {row['task_type']}\nTask: {row['goal']}\n"
            f"Observation: {row['observation']}\nInventory: {row['inventory']}\n"
            f"Recent observations: {'; '.join(row.get('recent_observations', [])[-5:])}\n"
            f"Recent actions: {'; '.join(row['history'][-5:])}\n"
            f"Admissible: {', '.join(row['admissible_actions'])}")


def weights(rows):
    counts = Counter(row["family"] for row in rows)
    return np.array([1 / counts[row["family"]] for row in rows])


class BenefitGate:
    """Three-class logistic head; estimate P(positive) - P(negative)."""

    def __init__(self, regularization=1.0):
        self.regularization = regularization
        self.threshold = 0.0

    def fit(self, rows):
        targets = np.array([r["gain"] for r in rows])
        self.constant = int(targets[0]) if len(set(targets.tolist())) == 1 else None
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True,
                                         min_df=1, max_features=30000)
        matrix = self.vectorizer.fit_transform([features(r) for r in rows])
        if self.constant is None:
            self.model = LogisticRegression(C=self.regularization, max_iter=2000,
                                            random_state=42)
            self.model.fit(matrix, targets, sample_weight=weights(rows) * len(rows) / weights(rows).sum())
        return self

    def scores(self, rows):
        if self.constant is not None:
            return np.full(len(rows), self.constant, dtype=float)
        probabilities = self.model.predict_proba(self.vectorizer.transform([features(r) for r in rows]))
        return probabilities @ self.model.classes_.astype(float)

    def activate(self, row):
        return bool(self.scores([row])[0] > self.threshold)


class SemanticRouter:
    """Textual relevance to the corresponding skill; no benefit labels in fitting."""

    def fit(self, rows):
        directory = Path("evals/alfworld/skills")
        self.skills = sorted(p.stem for p in directory.glob("*.txt"))
        documents = [(directory / f"{skill}.txt").read_text() for skill in self.skills]
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, max_features=30000)
        self.vectorizer.fit([features(r) for r in rows] + documents)
        self.documents = self.vectorizer.transform(documents)
        return self

    def scores(self, rows):
        similarity = cosine_similarity(self.vectorizer.transform([features(r) for r in rows]), self.documents)
        return np.array([similarity[i, self.skills.index(
            "pick_and_place" if r["task_type"] == "pick_two_and_place" else r["task_type"])]
                         for i, r in enumerate(rows)])


def metrics(rows, selected):
    selected = np.array(selected, dtype=bool)
    off = np.array([r["off"]["reward"] for r in rows])
    on = np.array([r["on"]["reward"] for r in rows])
    chosen = np.where(selected, on, off)
    gain = on - off
    sample_weights = weights(rows)
    return {"states": len(rows), "games": len({r["gamefile"] for r in rows}),
            "families": len({r["family"] for r in rows}),
            "selected_remaining_success_rate": float(chosen.mean()),
            "family_balanced_remaining_success_rate": float(np.average(chosen, weights=sample_weights)),
            "gain_vs_off": float((chosen - off).mean()),
            "activation_rate": float(selected.mean()),
            "negative_transfer_rate": float((selected & (gain < 0)).mean()),
            "harmful_activation_fraction": float(selected[gain < 0].mean()) if np.any(gain < 0) else None,
            "positive_transfer_retention": float(selected[gain > 0].mean()) if np.any(gain > 0) else None,
            "gain_counts": {str(value): int(np.sum(gain == value)) for value in (-1, 0, 1)}}


def choose_threshold(rows, scores):
    # Tune only. Strict > yields conservative handling of zero predicted gain.
    thresholds = sorted({float(np.min(scores)) - 1e-6, 0.0, *map(float, scores)})
    def objective(threshold):
        result = metrics(rows, scores > threshold)
        return (result["family_balanced_remaining_success_rate"],
                -result["activation_rate"], -result["negative_transfer_rate"])
    threshold = max(thresholds, key=objective)
    return threshold, objective(threshold)


def validate_outcomes(rows, plan):
    validate_splits(plan["records"])
    expected = {r["gamefile"]: r for r in plan["records"] if r["split"] != "sealed_test"}
    if {r["gamefile"] for r in rows} != set(expected):
        raise ValueError("Incomplete discovery/counterfactual coverage; refuse training")
    ids = set()
    for row in rows:
        if row["id"] in ids:
            raise ValueError("Duplicate state")
        ids.add(row["id"])
        source = expected[row["gamefile"]]
        if any(row[key] != source[key] for key in ("split", "scene", "family", "task_type")):
            raise ValueError("Outcome does not match predeclared split")
        if row["gain"] != row["on"]["reward"] - row["off"]["reward"]:
            raise ValueError("Gain label is not the measured return difference")
        if row["on"]["start"] != row["off"]["start"] or row["snapshot"] != row["on"]["start"]:
            raise ValueError("Counterfactuals are not from the same replayed state")
        if row["on"]["remaining_budget"] != row["off"]["remaining_budget"]:
            raise ValueError("Counterfactual budgets differ")
        if row["on"]["remaining_budget"] != 50 - len(row["replay"]):
            raise ValueError("Remaining budget ignores prefix actions")
        if row["on"]["steps_log"][0]["input_ids_sha256"] != row["off"]["steps_log"][0]["input_ids_sha256"]:
            raise ValueError("On/Off initial prompts differ")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()
    root = Path(args.output_dir)
    output = root / "gates"
    if output.exists():
        raise FileExistsError("Internal validation is a one-pass report; choose a new experiment for changes")
    completion = json.loads((root / "collection_completion.json").read_text())
    if not completion or any(job["exit_code"] for job in completion):
        raise ValueError("Collection did not complete cleanly")
    plan = json.loads((root / "plan.json").read_text())
    rows = [json.loads(line) for path in sorted((root / "workers").glob("*/outcomes.jsonl"))
            for line in path.read_text().splitlines() if line.strip()]
    validate_outcomes(rows, plan)
    kept, owner, duplicates = [], {}, []
    order = {"train": 0, "tune": 1, "validation": 2}
    for row in sorted(rows, key=lambda r: (order[r["split"]], r["id"])):
        signature = digest(features(row))
        if signature in owner and owner[signature] != row["split"]:
            duplicates.append({"id": row["id"], "split": row["split"],
                               "matches_split": owner[signature]})
            continue
        owner[signature] = row["split"]
        kept.append(row)
    rows = kept
    splits = {split: [r for r in rows if r["split"] == split]
              for split in ("train", "tune", "validation")}
    train, tune, validation = (splits[s] for s in ("train", "tune", "validation"))
    if not all(splits.values()):
        raise ValueError("All three internal splits are required")
    candidates = []
    for value in (.1, 1., 10.):
        gate = BenefitGate(value).fit(train)
        gate.threshold, objective = choose_threshold(tune, gate.scores(tune))
        candidates.append((objective, -value, gate))
    _, _, benefit = max(candidates, key=lambda item: item[:2])
    semantic = SemanticRouter().fit(train)
    semantic.threshold, _ = choose_threshold(tune, semantic.scores(tune))
    means = {skill: float(np.average([r["gain"] for r in train if r["task_type"] == skill],
                                    weights=weights([r for r in train if r["task_type"] == skill])))
             for skill in sorted({r["task_type"] for r in train})}
    # Fixed training-only task-type baseline; never choose types from validation returns.
    type_decisions = {skill: value > 0 for skill, value in means.items()}
    selected = {"always_off": [False] * len(validation), "always_on": [True] * len(validation),
                "task_type_gate": [type_decisions.get(r["task_type"], False) for r in validation],
                "semantic_router": semantic.scores(validation) > semantic.threshold,
                "benefit_gate": benefit.scores(validation) > benefit.threshold,
                "retrospective_oracle": [r["gain"] > 0 for r in validation]}
    report = {"scope": "one decision at each fixed state, then fixed-mode continuation; not dynamic-gate full episodes",
              "input_sha256": digest(rows), "split_sizes": {s: len(v) for s, v in splits.items()},
              "removed_cross_split_feature_duplicates": duplicates,
              "split_gain_counts": {s: dict(Counter(r["gain"] for r in v)) for s, v in splits.items()},
              "task_type_training_mean_gain": means,
              "benefit_C": benefit.regularization, "benefit_threshold": benefit.threshold,
              "semantic_threshold": semantic.threshold,
              "selection": "train fits models; tune selects C/thresholds; validation never used to choose settings",
              "sealed_test": "not collected, trained, tuned, or evaluated",
              "results": {name: metrics(validation, active) for name, active in selected.items()}}
    for name, active in selected.items():
        report["results"][name]["per_task_type"] = {
            skill: metrics([r for r in validation if r["task_type"] == skill],
                           [a for r, a in zip(validation, active) if r["task_type"] == skill])
            for skill in sorted(means) if any(r["task_type"] == skill for r in validation)}
    output.mkdir()
    for name, model in (("benefit_gate", benefit), ("semantic_router", semantic), ("task_type_gate", type_decisions)):
        with (output / f"{name}.pkl").open("wb") as handle:
            pickle.dump(model, handle)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    with (output / "validation_predictions.jsonl").open("w") as handle:
        for i, row in enumerate(validation):
            handle.write(json.dumps({"id": row["id"], "gain": row["gain"],
                                     "decisions": {key: bool(value[i]) for key, value in selected.items()}}) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    # Keep pickle class references importable rather than bound to __main__.
    from evals.boundary_gate.stage2a_train import main as module_main
    module_main()
