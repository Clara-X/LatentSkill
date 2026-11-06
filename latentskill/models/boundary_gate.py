"""Lightweight text gate; importing this module does not load the LLM."""

import pickle
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def state_text(row):
    """Only pre-decision observations; never labels, rewards, IDs, or oracle actions."""
    return (
        f"Skill: {row['skill']}\nTask: {row['goal']}\n"
        f"Observation: {row['observation']}\nInventory: {row['inventory']}\n"
        f"Recent actions: {'; '.join(row.get('history', []))}"
    )


class BoundaryGate:
    """Per-skill TF-IDF + logistic heads with a shared activation threshold."""

    def __init__(self, threshold=0.5, seed=42):
        self.threshold = threshold
        self.seed = seed
        self.heads = {}

    def fit(self, rows, labels):
        for skill in sorted({row["skill"] for row in rows}):
            indices = [i for i, row in enumerate(rows) if row["skill"] == skill]
            target = np.asarray(labels)[indices]
            if len(set(target.tolist())) < 2:
                # Conservative and explicit fallback when advantage labels have one class.
                self.heads[skill] = float(target[0])
                continue
            model = Pipeline([
                ("text", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
                ("classifier", LogisticRegression(C=10.0, random_state=self.seed,
                                                  max_iter=1000)),
            ])
            model.fit([state_text(rows[i]) for i in indices], target)
            self.heads[skill] = model
        return self

    def probability(self, row):
        head = self.heads.get(row["skill"], 0.0)
        return head if isinstance(head, float) else float(
            head.predict_proba([state_text(row)])[0, 1]
        )

    def activate(self, row):
        return row["skill"] in self.heads and self.probability(row) >= self.threshold

    def select_adapter(self, row, adapter_state):
        """Pass the result directly as backbone.generate(adapter_state=...)."""
        return adapter_state if self.activate(row) else None

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with Path(path).open("wb") as handle:
            pickle.dump(self, handle)

    @staticmethod
    def load(path):
        # Only load locally created/trusted gate files: pickle executes Python code.
        with Path(path).open("rb") as handle:
            return pickle.load(handle)
