"""Frozen native-policy rollouts and replay-verified same-state counterfactuals."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace

from evals.boundary_gate.stage2a_data import digest, validate_splits


def policy_prompt(goal, observation, admissible, history):
    from evals.alfworld.prompts import ALFWORLD_TEMPLATE, ALFWORLD_TEMPLATE_NO_HIS
    arguments = dict(task_description=goal, current_observation=observation,
                     admissible_actions=", ".join(admissible))
    if not history:
        return ALFWORLD_TEMPLATE_NO_HIS.format(**arguments).lstrip("\n")
    recent = history[-5:]
    text = "\n".join(f"[Observation {i+1}: '{obs[:300]}', Action {i+1}: '{action}']"
                     for i, (obs, action) in enumerate(recent))
    return ALFWORLD_TEMPLATE.format(**arguments, step_count=len(history),
                                   history_length=len(recent), action_history=text,
                                   current_step=len(history) + 1).lstrip("\n")


def parse_action(output, admissible):
    matches = re.findall(r"<action>(.*?)</action>", output, re.IGNORECASE | re.DOTALL)
    parsed = matches[-1].strip() if matches else output[-30:].strip()
    norm = lambda value: re.sub(r"\s+", " ", value).strip().lower()
    lookup = {norm(action): action for action in admissible}
    legal = bool(matches) and norm(parsed) in lookup
    action = lookup[norm(parsed)] if legal else (parsed if matches and parsed else "look")
    return action, bool(matches), legal


def replay(gamefile, actions, expert=False):
    from evals.boundary_gate.prepare import start_game
    env = start_game(gamefile, expert=expert)
    try:
        state = env.reset()
        initial = state["feedback"]
        history, checks = [], []
        for action in actions:
            previous = state["feedback"]
            state, _, done = env.step(action)
            history.append((previous, action))
            checks.append([state["feedback"], list(state["admissible_commands"]), bool(state["won"])])
            if done:
                raise ValueError("Candidate replay reaches a terminal state")
        observation = state["feedback"]
        admissible = list(state["admissible_commands"])
        won = bool(state["won"])
        # A read-only observation command, applied in BOTH branches; not a policy step.
        inventory_state, _, _ = env.step("inventory")
        snapshot = {"observation": observation, "inventory": inventory_state["feedback"],
                    "admissible": admissible, "history": history, "won": won,
                    "replay_observations_sha256": digest([initial, checks]),
                    "goal": re.search(r"Your task is to:\s*([^\n]+)", initial).group(1).strip()}
        snapshot["state_sha256"] = digest(snapshot)
        return env, snapshot
    except Exception:
        env.close()
        raise


class FrozenPolicy:
    def __init__(self, checkpoint, seed=42):
        import torch
        from evals.alfworld.evaluate import load_model, load_skill_contexts
        from evals.boundary_gate.diagnose import adapter_summary
        from latentskill.models.lora_ops import freeze_adapter_state
        torch.manual_seed(seed)
        self.seed = seed
        self.net, metalora, self.tokenizer, self.device, self.cfg, self.backbone = load_model(
            SimpleNamespace(device="cuda", config_name="models/qwen3_8b", raw_base=False,
                            checkpoint=checkpoint))
        self.net.eval().requires_grad_(False)
        self.backbone.eval().requires_grad_(False)
        freeze_adapter_state(metalora)
        self.adapters, audits = {}, {}
        with torch.inference_mode():
            for key, text in load_skill_contexts("evals/alfworld/skills").items():
                encoded = self.tokenizer(text, return_tensors="pt", truncation=True,
                                         max_length=4096).to(self.device)
                state = self.net.build_adapter_state(encoded.input_ids, encoded.attention_mask, metalora)
                freeze_adapter_state(state)
                self.adapters[key] = state
                audits[key] = adapter_summary(state)
        self.audit = {"adapters": audits, "trainable_parameters": sum(
            p.numel() for p in self.net.parameters() if p.requires_grad),
            "scale": self.net.scale, "rank": self.cfg.model.lora_r,
            "precision": "native loader; no additional autocast or weight conversion",
            "chat_template_sha256": hashlib.sha256(self.tokenizer.chat_template.encode()).hexdigest(),
            "backbone_dtype": str(next(self.backbone.parameters()).dtype)}

    def action(self, skill, mode, goal, observation, admissible, history):
        import torch
        prompt = policy_prompt(goal, observation, admissible, history)
        encoded = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True,
            tokenize=True, return_tensors="pt", max_length=4096, truncation=True,
            return_dict=True, padding=False, enable_thinking=True).to(self.device)
        key = "pick_and_place" if skill == "pick_two_and_place" else skill
        kwargs = dict(input_ids=encoded.input_ids, attention_mask=encoded.attention_mask,
                      max_new_tokens=4096, pad_token_id=self.tokenizer.pad_token_id,
                      eos_token_id=self.tokenizer.eos_token_id, do_sample=False, ignore_mem_token=True)
        if mode == "on":
            kwargs["adapter_state"] = self.adapters[key]
        with torch.inference_mode():
            output = self.backbone.generate(**kwargs)
        suffix = output[0, encoded.input_ids.shape[1]:]
        text = self.tokenizer.decode(suffix, skip_special_tokens=True).strip()
        action, valid, legal = parse_action(text, admissible)
        return {"output": text, "final_action": action, "is_parse_valid": valid,
                "in_admissible": legal, "obs": observation, "admissible_actions": admissible,
                "prompt": prompt, "input_ids_sha256": digest(encoded.input_ids[0].tolist()),
                "generated_tokens": len(suffix), "hit_token_limit": len(suffix) >= 4096}


def rollout(policy, game, actions, mode, expected=None, progress=None):
    import torch
    torch.manual_seed(policy.seed)
    env, snapshot = replay(game["gamefile"], actions)
    try:
        if expected is not None and snapshot != expected:
            raise ValueError("Counterfactual replay mismatch; no label may be emitted")
        history = list(snapshot["history"])
        observation, admissible = snapshot["observation"], snapshot["admissible"]
        steps, won = [], False
        budget = 50 - len(actions)
        if budget <= 0:
            raise ValueError("Candidate has no remaining policy budget")
        for _ in range(budget):
            record = policy.action(game["task_type"], mode, snapshot["goal"], observation,
                                   admissible, history)
            state, _, done = env.step(record["final_action"])
            won = bool(state["won"])
            record.update(step=len(history) + 1, next_obs=state["feedback"], won=won)
            steps.append(record)
            if progress:
                progress(mode, len(history) + 1, won)
            history.append((observation, record["final_action"]))
            observation, admissible = state["feedback"], list(state["admissible_commands"])
            if done:
                break
        return {"reward": int(won), "won": won, "steps_log": steps,
                "remaining_budget": budget, "start": snapshot}
    finally:
        env.close()


def propose(game, trajectories, limit=4):
    """Heuristics nominate states only; they NEVER supply the benefit target."""
    pool = []
    def add(actions, kind):
        if len(actions) < 48:
            pool.append({"actions": list(actions), "proposal": kind})
    off, on = trajectories["off"]["steps_log"], trajectories["on"]["steps_log"]
    for i, (a, b) in enumerate(zip(off, on)):
        if a["final_action"] != b["final_action"]:
            add([s["final_action"] for s in off[:i]], "behavior_divergence")
            break
    opportunities = []
    for source, trajectory in trajectories.items():
        prefix = []
        for step in trajectory["steps_log"]:
            commands = step["admissible_actions"]
            key = any(a.startswith(("heat ", "cool ", "clean ", "use ", "move ", "take "))
                      for a in commands)
            if key:
                opportunities.append((prefix.copy(), "key_action_opportunity:" + source))
            repeated = len(prefix) >= 3 and len(set(prefix[-3:])) == 1
            if not step["in_admissible"] or repeated:
                add(prefix + [step["final_action"]], "recovery:" + source)
                break
            prefix.append(step["final_action"])
    # Prefer real transformation opportunities over generic take/move opportunities.
    for source, trajectory in trajectories.items():
        for i, step in enumerate(trajectory["steps_log"]):
            if any(a.startswith(("heat ", "cool ", "clean ", "use "))
                   for a in step["admissible_actions"]):
                opportunities.insert(0, ([s["final_action"] for s in trajectory["steps_log"][:i]],
                                         "transform_opportunity:" + source))
                break
    if opportunities:
        actions, kind = opportunities[0]
        pool.insert(0, {"actions": actions, "proposal": kind})
        env, snapshot = replay(game["gamefile"], actions)
        try:
            moves = [a for a in snapshot["admissible"] if a.startswith("go to ")]
            if moves:
                # Wrong-location boundary nomination; return labels still come from rollouts.
                pool.insert(1, {"actions": actions + [moves[0]],
                                "proposal": "location_intervention"})
        finally:
            env.close()
    # Initial state is a genuine neutral/control candidate, not assigned a neutral label.
    add([], "initial_control")
    result, seen, kinds = [], set(), set()
    for candidate in pool:
        key = tuple(candidate["actions"])
        category = candidate["proposal"].split(":")[0]
        if key in seen or category in kinds:
            continue
        if len(candidate["actions"]) >= 48:
            continue
        result.append(candidate)
        seen.add(key)
        kinds.add(category)
        if len(result) == limit:
            break
    return result


def worker(args):
    root = Path(args.output_dir).resolve()
    plan = json.loads((root / "plan.json").read_text())
    records = [r for r in plan["records"] if r["split"] != "sealed_test"]
    games = records[args.worker_index::args.workers]
    output = root / "workers" / f"worker_{args.worker_index:02d}"
    output.mkdir(parents=True, exist_ok=True)
    if (output / "outcomes.jsonl").exists():
        raise FileExistsError("Refusing to append to an existing worker dataset")
    policy = FrozenPolicy(plan["checkpoint"], plan["seed"])
    (output / "model_audit.json").write_text(json.dumps(policy.audit, indent=2) + "\n")
    with (output / "outcomes.jsonl").open("x") as handle:
        for game_index, game in enumerate(games):
            def progress(mode, step, won):
                (output / "progress.json").write_text(json.dumps({
                    "game": game_index + 1, "games": len(games), "gamefile": game["gamefile"],
                    "phase": phase, "mode": mode, "step": step, "won": won}) + "\n")
            phase = "discovery"
            trajectories = {mode: rollout(policy, game, [], mode, progress=progress)
                            for mode in ("off", "on")}
            name = digest(game["gamefile"])[:16]
            (output / f"trajectory_{name}.json").write_text(json.dumps({
                "game": game, "trajectories": trajectories}, indent=2) + "\n")
            proposals = propose(game, trajectories, plan["states_per_game"])
            for index, candidate in enumerate(proposals):
                phase = f"counterfactual_{index+1}/{len(proposals)}"
                actions = candidate["actions"]
                env, snapshot = replay(game["gamefile"], actions)
                env.close()
                branch_order = ("off", "on") if index % 2 == 0 else ("on", "off")
                branches = {mode: rollout(policy, game, actions, mode, expected=snapshot,
                                         progress=progress) for mode in branch_order}
                if branches["off"]["steps_log"][0]["input_ids_sha256"] != branches["on"]["steps_log"][0]["input_ids_sha256"]:
                    raise ValueError("On/Off tokenized start prompts differ")
                gain = branches["on"]["reward"] - branches["off"]["reward"]
                row = dict(game, id=digest([game["gamefile"], actions]),
                           skill=game["task_type"], goal=snapshot["goal"],
                           observation=snapshot["observation"], inventory=snapshot["inventory"],
                           history=[action for _, action in snapshot["history"]],
                           recent_observations=[obs for obs, _ in snapshot["history"]][-5:],
                           admissible_actions=snapshot["admissible"], replay=actions,
                           game_sha256=hashlib.sha256(Path(game["gamefile"]).read_bytes()).hexdigest(),
                           snapshot=snapshot, proposal=candidate["proposal"],
                           off=branches["off"], on=branches["on"], gain=gain,
                           label={-1: "negative", 0: "neutral", 1: "positive"}[gain])
                handle.write(json.dumps(row) + "\n")
                handle.flush()
                print(game["split"], game["task_type"], phase, "gain", gain, flush=True)
    (output / "complete.json").write_text(json.dumps({"games": len(games)}) + "\n")


def launch(args):
    root = Path(args.output_dir).resolve()
    plan = json.loads((root / "plan.json").read_text())
    validate_splits(plan["records"])
    gpus = args.gpus.split(",")
    if len(set(gpus)) != len(gpus):
        raise ValueError("GPU indices must be unique")
    jobs = root / "jobs"
    jobs.mkdir(exist_ok=False)
    def run(item):
        index, gpu = item
        command = [sys.executable, "-u", "-m", __package__ + ".stage2a_collect", "worker",
                   "--output-dir", str(root), "--worker-index", str(index), "--workers", str(len(gpus))]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                   TOKENIZERS_PARALLELISM="false", MODEL_PATH=str(Path("models/Qwen3-8B").resolve()))
        with (jobs / f"gpu_{gpu}.log").open("w") as log:
            process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            status = {"pid": process.pid, "gpu": gpu, "command": command}
            path = jobs / f"gpu_{gpu}.json"
            path.write_text(json.dumps(status, indent=2) + "\n")
            status["exit_code"] = process.wait()
            path.write_text(json.dumps(status, indent=2) + "\n")
            return status
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        statuses = list(pool.map(run, enumerate(gpus)))
    (root / "collection_completion.json").write_text(json.dumps(statuses, indent=2) + "\n")
    if any(row["exit_code"] for row in statuses):
        raise RuntimeError("Collection failed; inspect worker logs; do not train on partial data")
    subprocess.run([sys.executable, "-m", __package__ + ".stage2a_train", "--output-dir", str(root)], check=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["launch", "worker"])
    p.add_argument("--output-dir", required=True)
    p.add_argument("--gpus", default="2,3,4,5,6,7")
    p.add_argument("--worker-index", type=int, default=0)
    p.add_argument("--workers", type=int, default=1)
    args = p.parse_args()
    {"launch": launch, "worker": worker}[args.command](args)


if __name__ == "__main__":
    main()
