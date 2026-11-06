"""Predeclare scene-isolated Train splits and audit observable data provenance."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import re


TASK_TYPES = ("heat", "cool", "clean", "pick_and_place", "pick_two_and_place",
              "look_at_obj_in_light")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def normalized(text):
    return re.sub(r"[^a-z0-9]+", " ", text.replace("\\n", "\n").lower()).strip()


def task_type(path):
    name = str(path)
    for skill in ("heat", "cool", "clean"):
        if f"pick_{skill}_then_place" in name:
            return skill
    if "look_at_obj_in_light" in name:
        return "look_at_obj_in_light"
    if "pick_two_obj_and_place" in name:
        return "pick_two_and_place"
    return "pick_and_place"


def provenance(repo):
    """Report positive matches without pretending missing identifiers prove absence."""
    files = [repo / "data/skill_ift/train.json", repo / "data/skill_pretrain/train.jsonl",
             repo / "data/skill_pretrain/val.jsonl"]
    audit, goals, identifiers = [], set(), set()
    skill_hashes = {p.stem: digest(normalized(p.read_text()))
                    for p in (repo / "evals/alfworld/skills").glob("*.txt")}
    for path in files:
        row = {"path": str(path), "available": path.exists()}
        if path.exists():
            raw = path.read_text()
            records = json.loads(raw) if path.suffix == ".json" else [
                json.loads(line) for line in raw.splitlines() if line.strip()]
            contexts = Counter()
            alf, ids = 0, 0
            for record in records:
                text = json.dumps(record)
                found = re.findall(r"trial_T\d{8}_\d{6}_\d+", text)
                identifiers.update(found)
                ids += bool(found)
                is_alf = "ALFRED" in text or "admissible" in text.lower()
                alf += is_alf
                context = record.get("context", record.get("text", ""))
                contexts[digest(normalized(context))] += 1
                for message in record.get("conversations", []):
                    if message.get("role") == "user":
                        match = re.search(r"Your task is to:\s*([^\n]+)", message.get("content", ""))
                        if match:
                            goals.add(normalized(match.group(1)))
            row.update(sha256=hashlib.sha256(path.read_bytes()).hexdigest(), records=len(records),
                       alfworld_style_records=alf, records_with_trial_ids=ids,
                       exact_normalized_skill_matches={key: contexts[value]
                                                       for key, value in skill_hashes.items()})
        audit.append(row)
    old_games = set()
    # Only read manifests/probe inputs, never use old outcome labels for sampling.
    for path in (repo / "outputs").rglob("*.json"):
        if path.name not in {"all.json", "plan.json"}:
            continue
        try:
            value = json.loads(path.read_text())
            values = value if isinstance(value, list) else value.get("games", [])
            old_games.update(p for p in values if isinstance(p, str) and p.endswith("game.tw-pddl"))
            if isinstance(value, dict):
                old_games.update(row["gamefile"] for row in value.get("records", [])
                                 if isinstance(row, dict) and row.get("split") != "sealed_test")
        except (ValueError, OSError):
            continue
    pairs = repo / "outputs/boundary_gate/pairs.jsonl"
    if pairs.exists():
        old_games.update(json.loads(line)["gamefile"] for line in pairs.read_text().splitlines())
    old_scenes = {Path(p).parent.parent.name.split("-")[-1] for p in old_games}
    return {"sources": audit, "skill_text_sha256": skill_hashes,
            "known_goal_count": len(goals), "known_trial_ids": sorted(identifiers),
            "previously_inspected_scenes": sorted(old_scenes),
            "provenance_status": "incomplete: no verified compiler-training game manifest; missing sources remain unknown",
            "claim": "gate splits isolated; backbone/compiler exposure is not certified absent"}, goals, identifiers, old_scenes


def validate_splits(records):
    for field in ("gamefile", "family", "scene"):
        owners = defaultdict(set)
        for row in records:
            owners[row[field]].add(row["split"])
        if any(len(parts) > 1 for parts in owners.values()):
            raise ValueError(f"Cross-split leakage through {field}")
    paths = [row["gamefile"] for row in records]
    if len(paths) != len(set(paths)):
        raise ValueError("Duplicate game")


def setup(args):
    repo = Path(__file__).resolve().parents[2]
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError("Choose a new Stage 2A directory")
    audit, known_goals, known_ids, old_scenes = provenance(repo)
    candidates = defaultdict(list)
    for path in sorted((repo / "alfworld_data/alfworld/json_2.1.1/train").glob("*/trial_*/game.tw-pddl")):
        if "Sliced" in str(path) or "movable" in str(path) or path.parent.name in known_ids:
            continue
        if not json.loads(path.read_text()).get("solvable", False):
            continue
        family = path.parent.parent.name
        candidates[task_type(path)].append({"gamefile": str(path), "family": family,
                                            "scene": family.split("-")[-1],
                                            "task_type": task_type(path)})
    scenes = sorted({r["scene"] for values in candidates.values() for r in values})
    counts = {"train": args.train_per_type, "tune": args.tune_per_type,
              "validation": args.validation_per_type, "sealed_test": args.test_per_type}
    protected = {"6", "11", "21"}  # Preserve the reserve declared before Stage 1.
    chosen = None
    for attempt in range(1000):
        rng = random.Random(args.seed + attempt)
        order = scenes.copy()
        rng.shuffle(order)
        n = len(order)
        groups = {"train": set(order[:int(n * .55)]),
                  "tune": set(order[int(n * .55):int(n * .70)]),
                  "validation": set(order[int(n * .70):int(n * .85)]),
                  "sealed_test": set(order[int(n * .85):]) - old_scenes}
        for split in ("train", "tune", "validation"):
            groups[split] -= protected
        groups["sealed_test"] = (groups["sealed_test"] | protected) - old_scenes
        records = []
        for split, count in counts.items():
            for skill in TASK_TYPES:
                pool = [r for r in candidates[skill] if r["scene"] in groups[split]]
                rng.shuffle(pool)
                selected, used_scenes, used_families = [], set(), set()
                # Prefer different scenes, then different task families.
                for unique_scene in (True, False):
                    for row in pool:
                        if row["family"] in used_families or (unique_scene and row["scene"] in used_scenes):
                            continue
                        if len(selected) >= count:
                            break
                        selected.append(dict(row, split=split))
                        used_families.add(row["family"])
                        used_scenes.add(row["scene"])
                records.extend(selected)
        diverse = all(len({r["scene"] for r in records if r["split"] == split
                           and r["task_type"] == skill}) >= min(count, 2)
                      for split, count in counts.items() for skill in TASK_TYPES)
        if len(records) == sum(counts.values()) * len(TASK_TYPES) and diverse:
            chosen = records
            break
    if chosen is None:
        raise RuntimeError("Cannot satisfy scene-isolated six-type sampling")
    validate_splits(chosen)
    # Overlap is an audit annotation, never a reward/benefit label.
    from evals.boundary_gate.prepare import start_game
    for row in chosen:
        if row["split"] == "sealed_test":
            continue
        env = start_game(row["gamefile"], expert=False)
        try:
            initial = env.reset()["feedback"]
            goal = re.search(r"Your task is to:\s*([^\n]+)", initial).group(1)
            row["goal_content_seen_in_available_ift"] = normalized(goal) in known_goals
        finally:
            env.close()
    output.mkdir(parents=True)
    audit["selected_known_goal_matches"] = dict(Counter(
        r["split"] for r in chosen if r.get("goal_content_seen_in_available_ift")))
    audit["known_trial_matches_selected"] = 0
    audit["sealed_test_policy"] = "manifest only; not reset, executed, or used for model selection"
    (output / "overlap_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    plan = {"seed": args.seed, "split_search_attempt": attempt, "records": chosen,
            "source": "ALFWorld train only", "counts_per_type": counts,
            "protected_scenes": sorted(protected),
            "checkpoint": args.checkpoint, "max_steps": 50, "max_new_tokens": 4096,
            "states_per_game": args.states_per_game,
            "label": "Delta R = binary terminal success On - Off under identical remaining 50-step budget",
            "selection": "C and activation threshold on tune only; validation evaluated once; sealed_test untouched",
            "feature_duplicate_policy": "keep first split in train/tune/validation order; remove exact cross-split feature duplicates without using rewards"}
    (output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps({"games": len(chosen), "counts": counts, "audit": audit}, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--checkpoint", default="checkpoints/latentskill_sft_qwen3_8b/checkpoint-epoch-10")
    p.add_argument("--train-per-type", type=int, default=4)
    p.add_argument("--tune-per-type", type=int, default=2)
    p.add_argument("--validation-per-type", type=int, default=2)
    p.add_argument("--test-per-type", type=int, default=2)
    p.add_argument("--states-per-game", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    setup(p.parse_args())


if __name__ == "__main__":
    main()
