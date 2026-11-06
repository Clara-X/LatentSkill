"""Collect real state pairs differing only in the agent's location."""

import argparse
import hashlib
import json
import random
import re
from pathlib import Path

import textworld
from alfworld.agents.environment.alfred_tw_env import (
    AlfredDemangler, AlfredExpert, AlfredInfos,
)


TASKS = {
    "heat": "pick_heat_then_place_in_recep",
    "cool": "pick_cool_then_place_in_recep",
    "clean": "pick_clean_then_place_in_recep",
}


def start_game(path, expert=True):
    extras = ["gamefile"]
    wrappers = [AlfredDemangler(shuffle=False), AlfredInfos]
    if expert:
        extras.append("expert_plan")
        wrappers.append(AlfredExpert("handcoded"))
    return textworld.start(
        str(path),
        request_infos=textworld.EnvInfos(
            won=True, admissible_commands=True, extras=extras,
        ),
        wrappers=wrappers,
    )


def collect_pair(path, skill, split, max_steps):
    env = start_game(path)
    try:
        state = env.reset()
        initial = state["feedback"]
        goal = re.search(r"Your task is to:\s*(.*)", initial).group(1).strip()
        replay = []
        for _ in range(max_steps):
            action = state["extra.expert_plan"][0]
            if action.startswith(skill + " "):
                break
            state, _, done = env.step(action)
            replay.append(action)
            if done:
                raise ValueError("Expert finished before target operation")
        else:
            raise ValueError("Expert did not reach target operation")
        operation = action
        device = operation.split(" with ", 1)[1]
        navigation = "go to " + device
        moves = sorted(a for a in state["admissible_commands"]
                       if a.startswith("go to ") and a != navigation)
        if not moves:
            raise ValueError("No location-only intervention available")
        pair_id = hashlib.sha256(str(path).encode()).hexdigest()[:16]
        rows = []
        for applicable in (True, False):
            # Replay both branches to avoid depending on expert-wrapper copy semantics.
            state = env.reset()
            for command in replay:
                state, _, _ = env.step(command)
            branch = list(replay)
            if not applicable:
                state, _, _ = env.step(moves[0])
                branch.append(moves[0])
            state, _, _ = env.step("look")
            observation = state["feedback"]
            state, _, _ = env.step("inventory")
            inventory = state["feedback"]
            commands = state["admissible_commands"]
            assert (operation in commands) == applicable
            expected = operation if applicable else navigation
            assert expected in commands
            rows.append({
                "id": f"{pair_id}:{int(applicable)}", "pair_id": pair_id,
                "split": split, "skill": skill, "gamefile": str(path),
                "goal": goal, "observation": observation, "inventory": inventory,
                "history": branch[-5:], "replay": branch,
                "candidates": [operation, navigation], "expected_action": expected,
                "applicable": int(applicable), "intervention": "agent_location",
            })
        assert rows[0]["inventory"] == rows[1]["inventory"]
        return rows
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="alfworld_data/alfworld/json_2.1.1")
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-games", type=int, default=4, help="Per skill")
    parser.add_argument("--val-games", type=int, default=2, help="Per skill")
    parser.add_argument("--test-games", type=int, default=2, help="Per skill")
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if min(args.train_games, args.val_games, args.test_games) < 1:
        parser.error("Each split needs at least one game per skill")
    random.seed(args.seed)
    rng = random.Random(args.seed)
    root = Path(args.data_root).resolve()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows, failures, used_families = [], [], set()
    # Keep all trials of a task family together and never reuse a family across splits.
    for split, source, count in [
        ("train", "train", args.train_games),
        ("val", "valid_seen", args.val_games),
        ("test", "valid_unseen", args.test_games),
    ]:
        paths = sorted((root / source).rglob("game.tw-pddl"))
        for skill, task in TASKS.items():
            candidates = [p for p in paths if p.parent.parent.name.startswith(task + "-")
                          and "Sliced" not in str(p) and "movable" not in str(p)]
            rng.shuffle(candidates)
            collected = 0
            for path in candidates:
                family = path.parent.parent.name
                if family in used_families:
                    continue
                try:
                    pair = collect_pair(path, skill, split, args.max_steps)
                except Exception as exc:
                    failures.append({"gamefile": str(path), "error": str(exc)})
                    continue
                rows.extend(pair)
                used_families.add(family)
                collected += 1
                print(f"[{split}/{skill}] {collected}/{count}: {family}", flush=True)
                if collected == count:
                    break
            if collected != count:
                raise RuntimeError(f"Insufficient {split}/{skill} pairs: {collected}/{count}")
    with output.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    output.with_suffix(".manifest.json").write_text(json.dumps({
        "seed": args.seed, "arguments": vars(args), "rows": len(rows),
        "failures": failures, "label_source": "simulator operation preconditions",
        "scope": "next-operation probe, not full-skill applicability",
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
