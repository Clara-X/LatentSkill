"""Compare frozen LoRA on/off at identical real ALFWorld snapshots."""

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from evals.alfworld.evaluate import load_model
from evals.boundary_gate.prepare import start_game
from latentskill.models.lora_ops import freeze_adapter_state


def prompt_for(row):
    return (
        "You are operating in ALFWorld. Choose the best NEXT action for the current "
        "state. Return only the action, without explanation.\n"
        f"Task: {row['goal']}\nObservation: {row['observation']}\n"
        f"Inventory: {row['inventory']}\n"
        f"Recent actions: {'; '.join(row['history'])}\n"
        "Candidate actions:\n" + "\n".join(row["candidates"])
    )


def probe_prefix(tokenizer, row):
    prefix = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt_for(row)}],
        tokenize=True, add_generation_prompt=True, enable_thinking=False,
    )
    # Complete the assistant's thinking prefix explicitly, matching the repository's
    # checkpoint tokenizer while avoiding unconstrained reasoning-length effects.
    prefix += tokenizer.encode("<think>\n\n</think>\n\n", add_special_tokens=False)
    return prefix


def candidate_scores(backbone, tokenizer, row, adapter, device):
    prefix = probe_prefix(tokenizer, row)
    scores = []
    for candidate in row["candidates"]:
        suffix = tokenizer.encode(candidate, add_special_tokens=False)
        ids = torch.tensor([prefix + suffix], device=device)
        logits = backbone(
            input_ids=ids, attention_mask=torch.ones_like(ids),
            adapter_state=adapter, ignore_mem_token=True, use_cache=False,
            logits_to_keep=len(suffix) + 1,
        ).logits[0, :-1].float()
        target = torch.tensor(suffix, device=device)
        # Mean token log probability avoids favoring the shorter navigation action.
        scores.append(float(-F.cross_entropy(logits, target)))
    return scores


def execute(row, action):
    env = start_game(row["gamefile"], expert=False)
    try:
        state = env.reset()
        for previous in row["replay"]:
            state, _, _ = env.step(previous)
        state, _, _ = env.step("look")
        assert state["feedback"] == row["observation"], "Snapshot replay diverged"
        state, _, _ = env.step("inventory")
        assert state["feedback"] == row["inventory"], "Inventory replay diverged"
        admissible = action in state["admissible_commands"]
        state, reward, done = env.step(action)
        return {
            "action": action, "admissible": admissible,
            "next_observation": state["feedback"], "environment_reward": float(reward),
            "won": bool(state["won"]), "done": bool(done),
            "reward": float(action == row["expected_action"]),
        }
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config-name", default="models/qwen3_8b")
    parser.add_argument("--skill-dir", default="evals/alfworld/skills")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this 8B-model pilot")
    torch.manual_seed(args.seed)
    rows = [json.loads(line) for line in Path(args.pairs).read_text().splitlines() if line]
    checkpoint = Path(args.checkpoint)
    for name in ("mem_tokens.pt", "metalora.pth"):
        if not (checkpoint / name).is_file():
            raise FileNotFoundError(checkpoint / name)
    net, metalora, tokenizer, device, cfg, backbone = load_model(SimpleNamespace(
        device=args.device, config_name=args.config_name, raw_base=False,
        checkpoint=args.checkpoint,
    ))
    net.eval().requires_grad_(False)
    freeze_adapter_state(metalora)
    skill_hashes = {}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive create prevents accidentally combining or replacing runs.
    with output.open("x") as handle, torch.inference_mode(), torch.autocast(
        device_type="cuda", dtype=torch.bfloat16,
    ):
        for skill in sorted({row["skill"] for row in rows}):
            skill_path = Path(args.skill_dir) / f"{skill}.txt"
            skill_text = skill_path.read_text()
            skill_hashes[skill] = hashlib.sha256(skill_text.encode()).hexdigest()
            encoded = tokenizer(skill_text, return_tensors="pt", truncation=True,
                                max_length=4096).to(device)
            adapter = net.build_adapter_state(
                encoded.input_ids, encoded.attention_mask, metalora,
            )
            freeze_adapter_state(adapter)
            for row in (row for row in rows if row["skill"] == skill):
                result = dict(row)
                for mode, state in (("off", None), ("on", adapter)):
                    scores = candidate_scores(backbone, tokenizer, row, state, device)
                    action = row["candidates"][max(range(len(scores)), key=scores.__getitem__)]
                    result[mode] = execute(row, action)
                    result[mode]["candidate_scores"] = scores
                result["gain"] = result["on"]["reward"] - result["off"]["reward"]
                handle.write(json.dumps(result) + "\n")
                handle.flush()
                print(f"{row['split']} {row['id']} {skill}: "
                      f"off={result['off']['reward']} on={result['on']['reward']}", flush=True)
            del adapter
    output.with_suffix(".manifest.json").write_text(json.dumps({
        "arguments": vars(args), "torch": torch.__version__,
        "device": torch.cuda.get_device_name(device), "seed": args.seed,
        "checkpoint_files": {p.name: {"bytes": p.stat().st_size,
                                      "mtime_ns": p.stat().st_mtime_ns}
                             for p in checkpoint.iterdir() if p.is_file()},
        "skill_sha256": skill_hashes, "config": str(cfg),
        "scoring": "mean-token-log-probability, two candidate actions",
        "reward": "correct next operation/navigation action (not episode success)",
        "frozen": True, "autocast": "bfloat16",
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
