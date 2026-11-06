"""Shard native ALFWorld evaluation across GPUs without batching model requests."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from evals.boundary_gate.diagnose import paired_summary


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def report(root):
    plan = json.loads((root / "plan.json").read_text())
    expected = set(plan["games"])
    by_mode = {}
    for mode in ("base", "on", "text"):
        rows = {}
        for path in sorted((root / mode).glob("shard_*/results_detail_unseen.jsonl")):
            for line in path.read_text().splitlines():
                row = json.loads(line)
                key = row["gamefile"]
                if key in rows:
                    raise ValueError(f"Duplicate {mode} game: {key}")
                rows[key] = row
        if set(rows) != expected:
            raise ValueError(f"Incomplete {mode}: {len(rows)}/{len(expected)} games")
        by_mode[mode] = rows
    matches = sum(
        by_mode["base"][key]["steps_log"][0]["input_ids_sha256"]
        == by_mode["on"][key]["steps_log"][0]["input_ids_sha256"]
        for key in expected
    )
    if matches != len(expected):
        raise ValueError("Base/On initial prompt hashes differ")
    result = {"overall": paired_summary(by_mode), "per_skill": {},
              "matching_initial_prompt_hashes": matches, "protocol": plan["protocol"]}
    for skill in sorted({row["task_type"] for row in by_mode["base"].values()}):
        subset = {mode: {key: row for key, row in rows.items() if row["task_type"] == skill}
                  for mode, rows in by_mode.items()}
        result["per_skill"][skill] = paired_summary(subset)
    write_json(root / "full_task_report.json", result)
    print(json.dumps(result, indent=2), flush=True)


def launch(args):
    import yaml
    from alfworld.agents.environment import get_environment

    repo = Path(__file__).resolve().parents[2]
    root = Path(args.output_dir).resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root / "manifests").mkdir()
    data = repo / "alfworld_data/alfworld"
    config = repo / "evals/alfworld/config_tw.yaml"
    skills = repo / "evals/alfworld/skills"
    checkpoint = (repo / args.checkpoint).resolve()
    model = repo / "models/Qwen3-8B"
    for path in (data, config, skills, checkpoint, model):
        if not path.exists():
            raise FileNotFoundError(path)
    env_vars = dict(os.environ, ALFWORLD_DATA=str(data), MODEL_PATH=str(model),
                    PYTHONPATH=str(repo), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                    TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1")
    os.environ["ALFWORLD_DATA"] = str(data)
    cfg = yaml.safe_load(config.read_text())
    environment = get_environment(cfg["env"]["type"])(cfg, train_eval="eval_out_of_distribution")
    games = sorted(str(Path(p).resolve()) for p in environment.game_files)
    gpus = args.gpus.split(",")
    if len(gpus) != len(set(gpus)) or len(gpus) > len(games):
        raise ValueError("Specify unique GPUs with at least one game per GPU")
    common = ["--checkpoint", str(checkpoint), "--config_name", "models/qwen3_8b",
              "--split", "unseen", "--alfworld_data", str(data),
              "--alfworld_config", str(config), "--skill_context_dir", str(skills),
              "--max_steps", "50", "--max_new_tokens", "4096",
              "--context_max_length", "4096", "--conversation_max_length", "4096",
              "--device", "cuda", "--seed", "42", "--record_prompt"]
    plan = {"games": games, "gpus": gpus, "python": sys.executable,
            "protocol": {"entrypoint": "evals.alfworld.evaluate", "common_arguments": common,
                         "execution": "native serial per GPU; disjoint game shards",
                         "scope": "full unseen evaluation; never gate-training data"},
            "skill_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in sorted(skills.glob("*.txt"))}}
    write_json(root / "plan.json", plan)

    def worker(item):
        shard, gpu = item
        manifest = root / "manifests" / f"shard_{shard:02d}.json"
        write_json(manifest, games[shard::len(gpus)])
        failures = []
        for mode, flag in (("base", "--zero_shot"), ("on", None), ("text", "--skill_incontext")):
            output = root / mode / f"shard_{shard:02d}"
            output.mkdir(parents=True)
            command = [sys.executable, "-u", "-m", "evals.alfworld.evaluate", *common,
                       "--games_manifest", str(manifest), "--output_dir", str(output)]
            if flag:
                command.append(flag)
            status = {"mode": mode, "gpu": gpu, "shard": shard, "command": command}
            with (output / "alfworld.log").open("w") as log:
                process = subprocess.Popen(command, cwd=repo,
                                           env=dict(env_vars, CUDA_VISIBLE_DEVICES=gpu),
                                           stdout=log, stderr=subprocess.STDOUT)
                status["pid"] = process.pid
                write_json(output / "job.json", status)
                status["exit_code"] = process.wait()
                write_json(output / "job.json", status)
            print(f"GPU {gpu} {mode} exit={status['exit_code']}", flush=True)
            if status["exit_code"]:
                failures.append(status)
        return failures

    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        failures = [failure for result in pool.map(worker, enumerate(gpus)) for failure in result]
    write_json(root / "completion.json", {"failures": failures})
    if failures:
        raise RuntimeError("Evaluation failures: inspect completion.json and per-shard logs")
    report(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["launch", "report"])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--checkpoint", default="checkpoints/latentskill_sft_qwen3_8b/checkpoint-epoch-10")
    args = parser.parse_args()
    if args.command == "launch":
        launch(args)
    else:
        report(Path(args.output_dir))


if __name__ == "__main__":
    main()
