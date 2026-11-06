"""Full-episode diagnostics through the repository's original ALFWorld entrypoint."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import time


SKILLS = {"heat": "pick_heat_then_place_in_recep", "cool": "pick_cool_then_place_in_recep",
          "clean": "pick_clean_then_place_in_recep"}


def restrict_games(available, manifest):
    requested = json.loads(Path(manifest).read_text())
    if not isinstance(requested, list) or not requested:
        raise ValueError("Manifest must be a nonempty JSON list of game paths")
    requested = [str(Path(p).resolve()) for p in requested]
    lookup = {str(Path(p).resolve()): p for p in available}
    if len(requested) != len(set(requested)):
        raise ValueError("Duplicate games in manifest")
    missing = set(requested) - lookup.keys()
    if missing:
        raise ValueError(f"Manifest games unavailable in selected environment split: {sorted(missing)}")
    return [lookup[p] for p in requested]


def setup(args):
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    excluded = set(args.reserve_scenes.split(","))
    paths = sorted(Path(args.data_root).resolve().glob("*/trial_*/game.tw-pddl"))
    selected = []
    for skill, task in SKILLS.items():
        candidates = [p for p in paths if p.parent.parent.name.startswith(task + "-")
                      and p.parent.parent.name.split("-")[-1] not in excluded
                      and "Sliced" not in str(p) and "movable" not in str(p)]
        rng.shuffle(candidates)
        families, games = set(), []
        for path in candidates:
            if path.parent.parent.name in families:
                continue
            if not json.loads(path.read_text()).get("solvable", False):
                continue
            games.append(str(path))
            families.add(path.parent.parent.name)
            if len(games) == args.per_skill:
                break
        if len(games) != args.per_skill:
            raise ValueError(f"Not enough {skill} games")
        (output / f"{skill}.json").write_text(json.dumps(games, indent=2) + "\n")
        selected.extend(games)
    plan = {
        "seed": args.seed, "split": "seen", "games": selected,
        "reserved_scenes": sorted(excluded), "purpose": "development only; not a final test",
        "scene_counts": dict(Counter(Path(p).parent.parent.name.split("-")[-1] for p in selected)),
        "protocol": {"entrypoint": "evals.alfworld.evaluate.main", "max_steps": 50,
                     "max_new_tokens": 512, "history_length": 5, "thinking": True,
                     "stored_weights": "float32", "autocast": "bfloat16",
                     "decoding": "greedy free generation; original action parser"},
    }
    (output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    (output / "all.json").write_text(json.dumps(selected, indent=2) + "\n")
    print(json.dumps(plan, indent=2))


def adapter_summary(tree):
    import torch
    digest = hashlib.sha256()
    count, elements, squared, trainable = 0, 0, 0.0, 0
    def visit(node):
        nonlocal count, elements, squared, trainable
        if isinstance(node, dict):
            for key in sorted(node, key=str):
                digest.update(str(key).encode())
                visit(node[key])
        elif torch.is_tensor(node):
            tensor = node.detach().float().cpu().contiguous()
            digest.update(tensor.numpy().tobytes())
            count += 1
            elements += tensor.numel()
            squared += float(tensor.square().sum())
            trainable += int(node.requires_grad)
        elif isinstance(node, (list, tuple)):
            for child in node:
                visit(child)
    visit(tree)
    return {"sha256": digest.hexdigest(), "tensor_count": count, "elements": elements,
            "l2_norm": squared ** 0.5, "trainable_tensors": trainable}


def run(args):
    import torch
    from evals.alfworld import evaluate
    from latentskill.models.lora_ops import freeze_adapter_state
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "results_detail_seen.jsonl").exists():
        raise FileExistsError("Choose a fresh output directory; results will not be overwritten")
    loader = evaluate.load_model
    audit = {"arguments": vars(args), "adapters": [], "generation_checks": [],
             "torch": torch.__version__, "device": torch.cuda.get_device_name(0)}
    def save_audit():
        (output / "adapter_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    def audited_loader(*pos, **kw):
        result = loader(*pos, **kw)
        net, metalora, tokenizer, device, cfg, backbone = result
        backbone.eval().requires_grad_(False)
        if net is not None:
            net.eval().requires_grad_(False)
            freeze_adapter_state(metalora)
            audit.update({"scale": net.scale, "rank": cfg.model.lora_r,
                          "checkpoint": args.checkpoint})
            build = net.build_adapter_state
            def audited_build(*a, **k):
                state = build(*a, **k)
                freeze_adapter_state(state)
                audit["adapters"].append(adapter_summary(state))
                save_audit()
                return state
            net.build_adapter_state = audited_build
            generate = backbone.generate
            checked = False
            def audited_generate(*a, **k):
                nonlocal checked
                state = k.get("adapter_state")
                if checked or state is None:
                    return generate(*a, **k)
                checked = True
                inputs = {name: k[name] for name in ("input_ids", "attention_mask")}
                with torch.inference_mode():
                    off = backbone(**inputs, adapter_state=None, ignore_mem_token=True,
                                   use_cache=False, logits_to_keep=1).logits.float()
                    on = backbone(**inputs, adapter_state=state, ignore_mem_token=True,
                                  use_cache=False, logits_to_keep=1).logits.float()
                    reference = generate(*a, **dict(k, adapter_state=None))
                    actual = generate(*a, **k)
                length = inputs["input_ids"].shape[-1]
                audit["generation_checks"].append({
                    "logit_max_abs_delta": float((on - off).abs().max()),
                    "logit_mean_abs_delta": float((on - off).abs().mean()),
                    "off_output": tokenizer.decode(reference[0, length:], skip_special_tokens=True),
                    "on_output": tokenizer.decode(actual[0, length:], skip_special_tokens=True),
                    "same_output_tokens": bool(torch.equal(reference, actual)),
                    "input_ids_sha256": hashlib.sha256(
                        json.dumps(inputs["input_ids"][0].tolist()).encode()).hexdigest(),
                    "chat_template_sha256": hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
                    "tokenized_prompt": tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=False),
                })
                save_audit()
                return actual
            backbone.generate = audited_generate
        if args.cache_linear_weights:
            from evals.boundary_gate.batched_diagnose import cache_linear_weights
            generate_with_audit = backbone.generate
            cache_ready = False
            def cached_generate(*a, **k):
                nonlocal cache_ready
                if not cache_ready:
                    cache_ready = True
                    inputs = {name: k[name] for name in ("input_ids", "attention_mask")}
                    inputs.update(ignore_mem_token=True, use_cache=False, logits_to_keep=1)
                    states = [None] + ([k["adapter_state"]] if k.get("adapter_state") is not None else [])
                    count, delta = cache_linear_weights(backbone, inputs, states)
                    audit.update(cached_linear_modules=count, linear_cache_max_logit_delta=delta)
                    save_audit()
                return generate_with_audit(*a, **k)
            backbone.generate = cached_generate
        save_audit()
        return result
    evaluate.load_model = audited_loader
    sys.argv = ["evals.alfworld.evaluate", "--games_manifest", args.manifest,
                "--checkpoint", args.checkpoint, "--split", "seen", "--seed", str(args.seed),
                "--max_steps", "50", "--max_new_tokens", str(args.max_new_tokens),
                "--output_dir", str(output), "--record_prompt"]
    if args.mode == "base":
        sys.argv.append("--zero_shot")
    elif args.mode == "text":
        sys.argv.append("--skill_incontext")
    if args.debug:
        sys.argv.extend(["--debug_prompt", "--debug_steps", "1"])
    # Keep original prompt construction, chat template, history, parser, and env loop.
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        evaluate.main()
    audit["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 1024 ** 3
    save_audit()


def paired_summary(by_mode):
    keys = set(by_mode["base"])
    if not keys or any(set(values) != keys for values in by_mode.values()):
        raise ValueError("All conditions must contain exactly the same nonempty game set")
    result = {}
    for mode, games in by_mode.items():
        values = list(games.values())
        steps = [step for row in values for step in row["steps_log"]]
        positives = sum(row["won"] and not by_mode["base"][game]["won"] for game, row in games.items())
        negatives = sum(not row["won"] and by_mode["base"][game]["won"] for game, row in games.items())
        result[mode] = {
            "games": len(values), "successes": sum(bool(row["won"]) for row in values),
            "success_rate": sum(bool(row["won"]) for row in values) / len(values),
            "beneficial_games": positives, "harmful_games": negatives,
            "positive_transfer_rate": positives / len(values),
            "negative_transfer_rate": negatives / len(values),
            "neutral_games": len(values) - positives - negatives,
            "mean_steps": sum(row["steps"] for row in values) / len(values),
            "invalid_action_rate": sum(not s["in_admissible"] for s in steps) / max(len(steps), 1),
            "token_limit_rate": sum(s.get("hit_token_limit", False) for s in steps) / max(len(steps), 1),
            "parse_failure_rate": sum(not s.get("is_parse_valid", False) for s in steps) / max(len(steps), 1),
            "token_limit_without_action_rate": sum(
                s.get("hit_token_limit", False) and not s.get("is_parse_valid", False)
                for s in steps) / max(len(steps), 1),
            "mean_generated_tokens": sum(s.get("generated_tokens", 0) for s in steps) / max(len(steps), 1),
            "context_cap_rate": sum(s.get("prompt_tokens", 0) >= 4096 for s in steps) / max(len(steps), 1),
            "valid_bare_actions_rejected": sum(
                not s.get("is_parse_valid", False) and bool(s.get("output", "").strip())
                and s["output"].strip().splitlines()[-1].strip() in s.get("admissible_actions", [])
                for s in steps),
        }
    off = by_mode["base"]
    on = by_mode["on"]
    result["oracle_base_on"] = {
        "successes": sum(bool(off[key]["won"] or on[key]["won"]) for key in keys),
        "games": len(keys),
        "success_rate": sum(bool(off[key]["won"] or on[key]["won"]) for key in keys) / len(keys),
    }
    result["oracle_all_three"] = {
        "successes": sum(any(rows[key]["won"] for rows in by_mode.values()) for key in keys),
        "games": len(keys),
        "success_rate": sum(any(rows[key]["won"] for rows in by_mode.values()) for key in keys) / len(keys),
    }
    return result


def report(args):
    root = Path(args.output_dir)
    by_mode = {mode: {} for mode in ("base", "on", "text")}
    for mode in by_mode:
        for path in (root / mode).rglob("results_detail_seen.jsonl"):
            for line in path.read_text().splitlines():
                row = json.loads(line)
                game = str(Path(row["gamefile"]).resolve())
                if game in by_mode[mode]:
                    raise ValueError(f"Duplicate {mode} result for {game}")
                by_mode[mode][game] = row
    expected = set(json.loads((root / "manifests/all.json").read_text()))
    if any(set(games) != expected for games in by_mode.values()):
        raise ValueError("Results do not cover the predeclared manifest exactly")
    summary = {"overall": paired_summary(by_mode), "per_skill": {}}
    for skill in SKILLS:
        summary["per_skill"][skill] = paired_summary({
            mode: {game: row for game, row in games.items() if row["task_type"] == skill}
            for mode, games in by_mode.items()
        })
    common_hashes = []
    divergence = []
    for game in sorted(expected):
        off, on = by_mode["base"][game], by_mode["on"][game]
        a, b = off["steps_log"], on["steps_log"]
        common_hashes.append(a[0]["input_ids_sha256"] == b[0]["input_ids_sha256"])
        for index, (left, right) in enumerate(zip(a, b)):
            if left["final_action"] != right["final_action"]:
                if left["input_ids_sha256"] != right["input_ids_sha256"]:
                    raise ValueError("First action divergence did not share an identical input")
                divergence.append({
                    "gamefile": game, "skill": off["task_type"], "step": index + 1,
                    "prefix_actions": [s["final_action"] for s in a[:index]],
                    "prompt": left["prompt"], "input_ids_sha256": left["input_ids_sha256"],
                    "off_action": left["final_action"], "on_action": right["final_action"],
                    "off_won": off["won"], "on_won": on["won"],
                    "gain": int(on["won"]) - int(off["won"]),
                    "scene": Path(game).parent.parent.name.split("-")[-1],
                    "label": "full remaining episode return after identical prefix",
                })
                break
    if not all(common_hashes):
        raise ValueError("Base and LoRA initial tokenized prompts differ")
    summary["protocol_check"] = {"matching_initial_prompt_hashes": sum(common_hashes),
                                 "games": len(common_hashes)}
    summary["first_divergence_states"] = {
        skill: {
            "states": sum(row["skill"] == skill for row in divergence),
            "beneficial": sum(row["skill"] == skill and row["gain"] > 0 for row in divergence),
            "harmful": sum(row["skill"] == skill and row["gain"] < 0 for row in divergence),
            "neutral": sum(row["skill"] == skill and row["gain"] == 0 for row in divergence),
        }
        for skill in SKILLS
    }
    probe = root / "probe_audit.jsonl"
    if probe.exists():
        probes = [json.loads(line) for line in probe.read_text().splitlines()]
        checks = [row[mode] for row in probes for mode in ("off", "on")]
        summary["old_probe_audit"] = {
            "states": len(probes), "mode_state_checks": len(checks),
            "mean_score_agrees_with_free_generation": sum(
                r["mean_winner"] == r["free_action"] for r in checks),
            "length_normalization_changes_winner": sum(
                r["mean_winner"] != r["sum_winner"] for r in checks),
            "candidate_order_changes_winner": sum(
                r["mean_winner"] != r["reversed_order_winner"] for r in checks),
            "free_generation_inadmissible": sum(not r["free_action_admissible"] for r in checks),
            "scope": "Six previously inspected states; diagnostic only, not held-out evidence",
        }
    summary["backend_parity"] = {
        p.stem: json.loads(p.read_text()).get("parity")
        for p in root.glob("service_*.json") if ".progress." not in p.name
    }
    summary["backend_controls"] = {
        p.stem: {key: json.loads(p.read_text()).get(key) for key in (
            "paired_execution", "zero_adapter_max_logit_delta", "linear_cache_max_logit_delta",
        )}
        for p in root.glob("service_*.json") if ".progress." not in p.name
    }
    serial_manifest = root / "manifests/serial_control.json"
    if serial_manifest.exists():
        wanted = set(json.loads(serial_manifest.read_text()))
        controls = {}
        for mode in ("base", "on"):
            path = root / "serial_control" / mode / "results_detail_seen.jsonl"
            controls[mode] = {
                str(Path(row["gamefile"]).resolve()): row
                for row in ([json.loads(line) for line in path.read_text().splitlines()] if path.exists() else [])
            }
        if all(set(rows) == wanted for rows in controls.values()):
            comparison = paired_summary(controls)
            comparison.pop("oracle_all_three")
            comparison["success_agreement_with_batched"] = {
                mode: sum(bool(row["won"]) == bool(by_mode[mode][game]["won"])
                          for game, row in rows.items())
                for mode, rows in controls.items()
            }
            summary["serial_control"] = comparison
        else:
            summary["serial_control"] = {"status": "pending", "expected_games": len(wanted)}
    budget = root / "budget_audit.jsonl"
    if budget.exists():
        records = [json.loads(line) for line in budget.read_text().splitlines()]
        summary["budget_audit"] = {
            mode: {"initial_states": len(data),
                   "action_tags_at_512": sum(bool(r["original_512_action_tags"]) for r in data),
                   "action_tags_at_2048": sum(bool(r["action_tags"]) for r in data),
                   "max_tokens_used": max(r["generated_tokens"] for r in data)}
            for mode in ("off", "on") if (data := [r for r in records if r["mode"] == mode])
        }
    (root / "full_task_report.json").write_text(json.dumps(summary, indent=2) + "\n")
    (root / "first_divergence_states.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in divergence))
    print(json.dumps({key: value for key, value in summary.items()
                      if key != "backend_parity"}, indent=2))


def launch(args):
    root = Path(args.output_dir)
    gpus = [int(value) for value in args.gpus.split(",")]
    if len(set(gpus)) != len(gpus):
        raise ValueError("GPU list must be unique")
    jobs = [("on", skill) for skill in SKILLS]
    jobs += [(mode, skill) for skill in SKILLS for mode in ("base", "text")]
    pending, active, completed = list(jobs), {}, []
    status = root / "jobs.json"
    if status.exists():
        raise FileExistsError("jobs.json already exists; inspect the existing run")
    def save():
        status.write_text(json.dumps({"pending": pending, "active": [
            {"gpu": gpu, "mode": data[1], "skill": data[2], "pid": data[0].pid}
            for gpu, data in active.items()], "completed": completed}, indent=2) + "\n")
    while pending or active:
        for gpu in gpus:
            if gpu in active or not pending:
                continue
            mode, skill = pending.pop(0)
            output = root / mode / skill
            output.mkdir(parents=True, exist_ok=True)
            handle = (output / "run.log").open("x")
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="4",
                       MKL_NUM_THREADS="4", TOKENIZERS_PARALLELISM="false")
            command = [sys.executable, "-u", "-m", "evals.boundary_gate.diagnose", "run",
                       "--mode", mode, "--manifest", str(root / "manifests" / f"{skill}.json"),
                       "--output-dir", str(output)]
            process = subprocess.Popen(command, env=env, stdout=handle, stderr=subprocess.STDOUT)
            active[gpu] = (process, mode, skill, handle, time.time())
            print(f"START gpu={gpu} {mode}/{skill} pid={process.pid}", flush=True)
        for gpu, (process, mode, skill, handle, started) in list(active.items()):
            code = process.poll()
            if code is not None:
                handle.close()
                completed.append({"gpu": gpu, "mode": mode, "skill": skill,
                                  "exit_code": code, "seconds": time.time() - started})
                del active[gpu]
                print(f"DONE gpu={gpu} {mode}/{skill} exit={code}", flush=True)
        save()
        if pending or active:
            time.sleep(2)
    if any(job["exit_code"] for job in completed):
        raise RuntimeError("Some evaluation jobs failed; inspect jobs.json and individual run.log files")
    report(argparse.Namespace(output_dir=str(root)))


def audit_probe(args):
    """Measure the old probe's scoring and free generation on identical inputs."""
    from types import SimpleNamespace
    import torch
    from evals.alfworld.evaluate import load_model
    from evals.boundary_gate.collect import candidate_scores, execute, probe_prefix
    from latentskill.models.lora_ops import freeze_adapter_state
    rows = [json.loads(line) for line in Path(args.input).read_text().splitlines()]
    net, metalora, tokenizer, device, cfg, backbone = load_model(SimpleNamespace(
        device="cuda:0", config_name="models/qwen3_8b", raw_base=False,
        checkpoint=args.checkpoint,
    ))
    net.eval().requires_grad_(False)
    freeze_adapter_state(metalora)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle, torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for skill in SKILLS:
            pool = [r for r in rows if r["skill"] == skill and r["split"] == "test"]
            anchor = next((r for r in pool if r.get("gain", 0)), pool[0])
            probes = [r for r in pool if r["pair_id"] == anchor["pair_id"]]
            text = (Path("evals/alfworld/skills") / f"{skill}.txt").read_text()
            evidence = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096).to(device)
            adapter = net.build_adapter_state(evidence.input_ids, evidence.attention_mask, metalora)
            for row in probes:
                record = {"id": row["id"], "skill": skill, "applicable": row["applicable"],
                          "expected_action": row["expected_action"], "candidates": row["candidates"]}
                lengths = [len(tokenizer.encode(a, add_special_tokens=False)) for a in row["candidates"]]
                for mode, state in (("off", None), ("on", adapter)):
                    scores = candidate_scores(backbone, tokenizer, row, state, device)
                    reverse = dict(row, candidates=list(reversed(row["candidates"])))
                    reverse_scores = list(reversed(candidate_scores(
                        backbone, tokenizer, reverse, state, device)))
                    ids = torch.tensor([probe_prefix(tokenizer, row)], device=device)
                    generated = backbone.generate(
                        input_ids=ids, attention_mask=torch.ones_like(ids), adapter_state=state,
                        ignore_mem_token=True, do_sample=False, max_new_tokens=128,
                        pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                    )[0, ids.shape[-1]:]
                    text = tokenizer.decode(generated, skip_special_tokens=True).strip()
                    matches = re.findall(r"<action>(.*?)</action>", text, flags=re.S)
                    action = matches[-1].strip() if matches else text.splitlines()[0].strip()
                    result = execute(row, action)
                    sums = [p * length for p, length in zip(scores, lengths)]
                    record[mode] = {
                        "mean_log_probs": scores, "sum_log_probs": sums,
                        "reversed_order_mean_log_probs": reverse_scores, "token_lengths": lengths,
                        "mean_winner": row["candidates"][max(range(2), key=scores.__getitem__)],
                        "sum_winner": row["candidates"][max(range(2), key=sums.__getitem__)],
                        "reversed_order_winner": row["candidates"][max(range(2), key=reverse_scores.__getitem__)],
                        "free_output": text, "free_action": action,
                        "free_action_admissible": result["admissible"],
                        "free_action_in_candidates": action in row["candidates"],
                        "free_hit_limit": len(generated) >= 128,
                        "interpretation": "Admissibility is not full-task success; alternatives are not marked wrong",
                    }
                handle.write(json.dumps(record) + "\n")
                handle.flush()
                print(row["id"], skill, "probe audit done", flush=True)


def audit_budget(args):
    """Check token-budget sensitivity at the already audited initial dev states."""
    from types import SimpleNamespace
    import torch
    from evals.alfworld.evaluate import load_model
    from latentskill.models.lora_ops import freeze_adapter_state
    torch.manual_seed(42)
    net, metalora, tok, device, cfg, backbone = load_model(SimpleNamespace(
        device="cuda:0", config_name="models/qwen3_8b", raw_base=False,
        checkpoint=args.checkpoint,
    ))
    net.eval().requires_grad_(False)
    freeze_adapter_state(metalora)
    with Path(args.output).open("x") as handle, torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for skill in SKILLS:
            path = Path(args.serial_root) / "on" / skill / "adapter_audit.json"
            check = json.loads(path.read_text())["generation_checks"][0]
            prefix = tok.encode(check["tokenized_prompt"], add_special_tokens=False)
            if hashlib.sha256(json.dumps(prefix).encode()).hexdigest() != check["input_ids_sha256"]:
                raise ValueError("Initial audit prompt tokenization changed")
            evidence = tok((Path("evals/alfworld/skills") / f"{skill}.txt").read_text().strip(),
                           return_tensors="pt", truncation=True, max_length=4096).to(device)
            adapter = net.build_adapter_state(evidence.input_ids, evidence.attention_mask, metalora)
            ids = torch.tensor([prefix], device=device)
            for mode, state in (("off", None), ("on", adapter)):
                started = time.time()
                generated = backbone.generate(
                    input_ids=ids, attention_mask=torch.ones_like(ids), adapter_state=state,
                    ignore_mem_token=True, do_sample=False, max_new_tokens=args.budget,
                    pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
                )[0, len(prefix):]
                text = tok.decode(generated, skip_special_tokens=True).strip()
                original = check[f"{mode}_output"]
                record = {
                    "skill": skill, "mode": mode, "budget": args.budget,
                    "generated_tokens": len(generated), "hit_limit": len(generated) >= args.budget,
                    "output": text, "action_tags": re.findall(r"<action>(.*?)</action>", text, re.S),
                    "original_512_action_tags": re.findall(r"<action>(.*?)</action>", original, re.S),
                    "original_512_output": original, "seconds": time.time() - started,
                    "scope": "initial development state only; not full-task success",
                }
                handle.write(json.dumps(record) + "\n")
                handle.flush()
                print(skill, mode, 'tokens',len(generated),'actions',record['action_tags'],flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("setup")
    p.add_argument("--data-root", default="alfworld_data/alfworld/json_2.1.1/valid_seen")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--per-skill", type=int, default=10)
    p.add_argument("--reserve-scenes", default="6,11,21")
    p.add_argument("--seed", type=int, default=42)
    p = sub.add_parser("run")
    p.add_argument("--manifest", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--mode", choices=["base", "on", "text"], required=True)
    p.add_argument("--checkpoint", default="checkpoints/latentskill_sft_qwen3_8b/checkpoint-epoch-10")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--debug", action="store_true")
    p.add_argument("--cache-linear-weights", action="store_true")
    p = sub.add_parser("report")
    p.add_argument("--output-dir", required=True)
    p = sub.add_parser("launch")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--gpus", default="4,0,1,2,3")
    p = sub.add_parser("audit-probe")
    p.add_argument("--input", default="outputs/boundary_gate/outcomes.jsonl")
    p.add_argument("--output", required=True)
    p.add_argument("--checkpoint", default="checkpoints/latentskill_sft_qwen3_8b/checkpoint-epoch-10")
    p = sub.add_parser("audit-budget")
    p.add_argument("--serial-root", default="outputs/alfworld_stage1")
    p.add_argument("--output", required=True)
    p.add_argument("--budget", type=int, default=2048)
    p.add_argument("--checkpoint", default="checkpoints/latentskill_sft_qwen3_8b/checkpoint-epoch-10")
    args = parser.parse_args()
    {"setup": setup, "run": run, "report": report, "launch": launch,
     "audit-probe": audit_probe, "audit-budget": audit_budget}[args.command](args)


if __name__ == "__main__":
    main()
