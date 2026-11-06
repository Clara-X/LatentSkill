"""Batch model requests while retaining the original per-game ALFWorld evaluator.

Each CPU worker calls evaluate.main unchanged. Only load_model/generate are RPC
proxies; the GPU service groups requests sharing the same frozen adapter.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
from multiprocessing.connection import Client, Listener
import queue
import subprocess
import sys
import threading
import time
from types import SimpleNamespace


def ids_key(ids):
    return hashlib.sha256(json.dumps(ids).encode()).hexdigest()


def paired_batches(requests, batch_size):
    """Keep each game's Base/On rows in one kernel batch, in a fixed order."""
    if batch_size < 2:
        raise ValueError("Paired execution needs batch size >= 2")
    units = []
    games = sorted({key.split(":")[0] for key in requests})
    for game in games:
        pair = [requests[key] for mode in ("base", "on")
                if (key := f"{game}:{mode}") in requests]
        if pair:
            units.append(pair)
    units.extend([requests[key]] for key in sorted(requests) if key.endswith(":text"))
    batch = []
    for unit in units:
        if len(batch) + len(unit) > batch_size:
            yield batch
            batch = []
        batch.extend(unit)
    if batch:
        yield batch


def stack_adapters(trees):
    import torch
    if isinstance(trees[0], dict):
        return {key: stack_adapters([tree[key] for tree in trees]) for key in trees[0]}
    if trees[0] is None:
        return None
    return torch.cat(trees, dim=0)


def zero_adapter(tree):
    import torch
    if isinstance(tree, dict):
        return {key: zero_adapter(value) for key, value in tree.items()}
    return torch.zeros_like(tree) if torch.is_tensor(tree) else None


def cache_linear_weights(backbone, inputs, states):
    import torch
    before = [backbone(**inputs, adapter_state=state).logits.float().clone() for state in states]
    cached = 0
    for module in backbone.modules():
        if isinstance(module, torch.nn.Linear):
            module.to(dtype=torch.bfloat16)
            cached += 1
    after = [backbone(**inputs, adapter_state=state).logits.float() for state in states]
    delta = max(float((a - b).abs().max()) for a, b in zip(before, after))
    if delta != 0:
        raise ValueError(f"Cached autocast Linear weights changed logits: {delta}")
    return cached, delta


def server(args):
    import torch
    from omegaconf import OmegaConf
    from evals.alfworld.evaluate import load_model, load_skill_contexts
    from latentskill.models.lora_ops import freeze_adapter_state
    from evals.boundary_gate.diagnose import adapter_summary
    torch.manual_seed(42)
    net, metalora, tok, device, cfg, backbone = load_model(SimpleNamespace(
        device="cuda:0", config_name="models/qwen3_8b", raw_base=False,
        checkpoint=args.checkpoint,
    ))
    net.eval().requires_grad_(False)
    freeze_adapter_state(metalora)
    adapters, summaries = {}, {}
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for skill, text in load_skill_contexts("evals/alfworld/skills").items():
            evidence = tok(text, truncation=True, max_length=4096, return_tensors="pt").to(device)
            key = ids_key(evidence.input_ids[0].tolist())
            state = net.build_adapter_state(evidence.input_ids, evidence.attention_mask, metalora)
            freeze_adapter_state(state)
            adapters[key] = state
            summaries[skill] = {"key": key, **adapter_summary(state)}
    parity = None
    if args.parity_audit:
        reference = json.loads(Path(args.parity_audit).read_text())
        check = reference["generation_checks"][0]
        prefix = tok.encode(check["tokenized_prompt"], add_special_tokens=False)
        if ids_key(prefix) != check["input_ids_sha256"]:
            raise ValueError("Serial audit prompt did not round-trip through the shared tokenizer")
        skill = Path(args.parity_audit).parent.name
        # Row zero keeps the audited input; the second row exercises left padding.
        packed = torch.full((2, len(prefix)), tok.pad_token_id, dtype=torch.long, device=device)
        mask = torch.zeros_like(packed)
        packed[0] = torch.tensor(prefix, device=device)
        mask[0] = 1
        packed[1, -16:] = torch.tensor(prefix[-16:], device=device)
        mask[1, -16:] = 1
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            generated = backbone.generate(
                input_ids=packed, attention_mask=mask,
                adapter_state=adapters[summaries[skill]["key"]], ignore_mem_token=True,
                max_new_tokens=512, do_sample=False, pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
            )
        text = tok.decode(generated[0, len(prefix):], skip_special_tokens=True).strip()
        parity = {"serial_audit": args.parity_audit,
                  "input_hash_matches": True,
                  "batch_two_matches_serial_text": text == check["on_output"].strip(),
                  "batch_two_output": text, "serial_output": check["on_output"],
                  "adapter_hash_matches": summaries[skill]["sha256"] ==
                      reference["adapters"][["pick_and_place", "cool", "heat", "clean", "look_at_obj_in_light"].index(skill)]["sha256"]}
    zero = zero_adapter(next(iter(adapters.values())))
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        control_ids = torch.tensor([[100, 101, 102], [100, 101, 102]], device=device)
        control_args = dict(input_ids=control_ids, attention_mask=torch.ones_like(control_ids),
                            ignore_mem_token=True, use_cache=False, logits_to_keep=1)
        no_adapter = backbone(**control_args, adapter_state=None).logits.float()
        zero_logits = backbone(**control_args, adapter_state=stack_adapters([zero, zero])).logits.float()
        zero_delta = float((no_adapter - zero_logits).abs().max())
        if zero_delta != 0:
            raise ValueError(f"Zero-LoRA control differs from Base: {zero_delta}")
        # Autocast already rounds Linear weights to bf16 on every call. Cache
        # exactly that cast once, keeping embeddings/norms/residual streams fp32.
        # Verify both conditions before allowing the optimization into a run.
        cache_ids = torch.tensor([prefix if args.parity_audit else [100, 101, 102]], device=device)
        cache_args = dict(input_ids=cache_ids, attention_mask=torch.ones_like(cache_ids),
                          ignore_mem_token=True, use_cache=False, logits_to_keep=1)
        sample_adapter = next(iter(adapters.values()))
        cached, cache_delta = cache_linear_weights(backbone, cache_args, (None, sample_adapter))
    inbox = queue.Queue()
    listener = Listener(args.socket, family="AF_UNIX")
    def accept():
        while True:
            try:
                conn = listener.accept()
                def receive(connection):
                    try:
                        inbox.put((connection.recv(), connection))
                    except EOFError:
                        connection.close()
                threading.Thread(target=receive, args=(conn,), daemon=True).start()
            except OSError:
                return
    threading.Thread(target=accept, daemon=True).start()
    Path(args.meta).write_text(json.dumps({
        "chat_template": tok.chat_template, "config": OmegaConf.to_container(cfg, resolve=False),
        "scale": net.scale, "adapters": summaries, "checkpoint": args.checkpoint,
        "torch": torch.__version__, "batch_size": args.batch_size,
        "precision": "original float32-loaded parameters; cached bfloat16 Linear weights; fp32 embeddings/norms; bfloat16 autocast",
        "note": "Padding/batching can introduce numerical differences from serial greedy decoding",
        "parity": parity,
        "paired_execution": True, "zero_adapter_max_logit_delta": zero_delta,
        "cached_linear_modules": cached, "linear_cache_max_logit_delta": cache_delta,
    }, indent=2) + "\n")
    print("READY", args.socket, flush=True)
    calls, generated_tokens, batches, round_index = 0, 0, 0, 0
    active = {f"{index:02d}:{mode}" for index in range(args.game_count)
              for mode in ("base", "on", "text")}
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        while active:
            requests = {}
            while active - requests.keys():
                item = inbox.get()
                packet, conn = item
                client = packet["client_id"]
                if client not in active:
                    raise ValueError(f"Unexpected client {client}")
                if packet.get("kind") == "done":
                    active.remove(client)
                    conn.send({"ok": True})
                    conn.close()
                else:
                    if client in requests:
                        raise ValueError(f"Duplicate request from {client}")
                    requests[client] = item
            round_index += 1
            for batch in paired_batches(requests, args.batch_size):
                width = max(len(item[0]["ids"]) for item in batch)
                inputs = torch.full((len(batch), width), tok.pad_token_id,
                                    dtype=torch.long, device=device)
                mask = torch.zeros_like(inputs)
                for index, (item, _) in enumerate(batch):
                    length = len(item["ids"])
                    inputs[index, -length:] = torch.tensor(item["ids"], device=device)
                    mask[index, -length:] = 1
                adapter = stack_adapters([
                    adapters[item["adapter"]] if item["adapter"] else zero for item, _ in batch
                ])
                result = backbone.generate(
                    input_ids=inputs, attention_mask=mask, adapter_state=adapter,
                    ignore_mem_token=True, max_new_tokens=512, do_sample=False,
                    pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
                )[:, width:].cpu().tolist()
                for (item, conn), suffix in zip(batch, result):
                    if tok.eos_token_id in suffix:
                        suffix = suffix[:suffix.index(tok.eos_token_id) + 1]
                    conn.send({"ids": item["ids"] + suffix})
                    conn.close()
                    calls += 1
                    generated_tokens += len(suffix)
                batches += 1
                Path(args.meta).with_suffix(".progress.json").write_text(json.dumps({
                    "calls": calls, "batches": batches, "generated_tokens": generated_tokens,
                    "last_batch_size": len(batch), "active_workers": len(active), "round": round_index,
                    "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024 ** 3,
                }) + "\n")
    threading.Event().wait()


def worker(args):
    import torch
    from omegaconf import OmegaConf
    from transformers import AutoTokenizer
    from evals.alfworld import evaluate
    meta = json.loads(Path(args.meta).read_text())
    tok = AutoTokenizer.from_pretrained("models/Qwen3-8B", padding_side="left", use_fast=True)
    tok.add_tokens(["<RECON>", "<COMP>", "<NOTHING>"])
    tok.chat_template = meta["chat_template"]
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    class Backbone:
        def generate(self, **kwargs):
            if kwargs["input_ids"].shape[0] != 1 or kwargs["max_new_tokens"] != 512:
                raise ValueError("This transport is configured for official batch=1, 512-token requests")
            with Client(args.socket, family="AF_UNIX") as conn:
                conn.send({"ids": kwargs["input_ids"][0].tolist(),
                           "adapter": kwargs.get("adapter_state"), "client_id": args.client_id})
                result = conn.recv()
            if "error" in result:
                raise RuntimeError(result["error"])
            return torch.tensor([result["ids"]], dtype=torch.long)
    backbone = Backbone()
    keys = {data["key"] for data in meta["adapters"].values()}
    class Net:
        scale = meta["scale"]
        def __init__(self):
            self.backbone = backbone
        def build_adapter_state(self, evidence_ids, *unused):
            key = ids_key(evidence_ids[0].tolist())
            if key not in keys:
                raise ValueError("Worker skill tokenization differs from GPU service")
            return key
    def load_proxy(original_args, base_only=False):
        return (None if base_only else Net(), None, tok, torch.device("cpu"),
                OmegaConf.create(meta["config"]), backbone)
    evaluate.load_model = load_proxy
    sys.argv = ["evals.alfworld.evaluate", "--games_manifest", args.manifest,
                "--split", "seen", "--seed", "42", "--max_steps", "50",
                "--max_new_tokens", "512", "--output_dir", args.output_dir, "--record_prompt"]
    if args.mode == "base":
        sys.argv.append("--zero_shot")
    elif args.mode == "text":
        sys.argv.append("--skill_incontext")
    try:
        evaluate.main()
    finally:
        with Client(args.socket, family="AF_UNIX") as conn:
            conn.send({"kind": "done", "client_id": args.client_id})
            conn.recv()


def launch(args):
    root = Path(args.output_dir)
    services, workers = [], []
    gpu_list = args.gpus.split(",")
    if len(gpu_list) != 3:
        raise ValueError("Specify one GPU per heat/cool/clean service")
    try:
        for skill, gpu in zip(("heat", "cool", "clean"), gpu_list):
            meta = root / f"service_{skill}.json"
            socket = f"/tmp/latentskill_stage1_{os.getpid()}_{skill}.sock"
            log = (root / f"service_{skill}.log").open("x")
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, OMP_NUM_THREADS="4",
                       MKL_NUM_THREADS="4", TOKENIZERS_PARALLELISM="false")
            cmd = [sys.executable, "-u", "-m", "evals.boundary_gate.batched_diagnose", "server",
                   "--socket", socket, "--meta", str(meta), "--batch-size", str(args.batch_size),
                   "--game-count", str(len(json.loads((root / "manifests" / f"{skill}.json").read_text())))]
            serial_audit = Path("outputs/alfworld_stage1/on") / skill / "adapter_audit.json"
            if serial_audit.exists():
                cmd.extend(["--parity-audit", str(serial_audit)])
            process = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
            services.append((skill, process, meta, socket, log))
        while not all(meta.exists() for _, _, meta, _, _ in services):
            if any(process.poll() is not None for _, process, _, _, _ in services):
                raise RuntimeError("GPU service failed during startup; inspect service logs")
            time.sleep(2)
        for skill, process, meta, socket, log in services:
            games = json.loads((root / "manifests" / f"{skill}.json").read_text())
            for index, game in enumerate(games):
                manifest = root / "manifests" / f"{skill}_{index:02d}.json"
                manifest.write_text(json.dumps([game]) + "\n")
                for mode in ("base", "on", "text"):
                    output = root / mode / skill / f"game_{index:02d}"
                    output.mkdir(parents=True, exist_ok=True)
                    handle = (output / "run.log").open("x")
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1",
                               MKL_NUM_THREADS="1", TOKENIZERS_PARALLELISM="false")
                    command = [sys.executable, "-u", "-m", "evals.boundary_gate.batched_diagnose", "worker",
                               "--mode", mode, "--manifest", str(manifest), "--output-dir", str(output),
                               "--socket", socket, "--meta", str(meta),
                               "--client-id", f"{index:02d}:{mode}"]
                    child = subprocess.Popen(command, env=env, stdout=handle, stderr=subprocess.STDOUT)
                    workers.append((child, mode, skill, index, handle))
                    time.sleep(0.04)
        while any(process.poll() is None for process, *_ in workers):
            if any(process.poll() not in (None, 0) for process, *_ in workers):
                raise RuntimeError("An original-evaluator worker failed; inspect its run.log")
            if any(process.poll() is not None for _, process, _, _, _ in services):
                raise RuntimeError("GPU service exited before workers finished")
            (root / "batched_jobs.json").write_text(json.dumps([
                {"pid": process.pid, "mode": mode, "skill": skill, "game": index,
                 "exit_code": process.poll()} for process, mode, skill, index, _ in workers
            ], indent=2) + "\n")
            time.sleep(2)
        if any(process.returncode for process, *_ in workers):
            raise RuntimeError("Some original-evaluator workers failed; inspect per-game logs")
        from evals.boundary_gate.diagnose import report
        report(SimpleNamespace(output_dir=str(root)))
    finally:
        for process, _, _, _, log in workers:
            if process.poll() is None:
                process.terminate()
            process.wait()
            log.close()
        for _, process, _, socket, log in services:
            if process.poll() is None:
                process.terminate()
            process.wait()
            log.close()
            Path(socket).unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("server")
    p.add_argument("--socket", required=True)
    p.add_argument("--meta", required=True)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--game-count", type=int, default=10)
    p.add_argument("--parity-audit", default=None)
    p.add_argument("--checkpoint", default="checkpoints/latentskill_sft_qwen3_8b/checkpoint-epoch-10")
    p = sub.add_parser("worker")
    p.add_argument("--socket", required=True)
    p.add_argument("--meta", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--mode", choices=["base", "on", "text"], required=True)
    p.add_argument("--client-id", required=True)
    p = sub.add_parser("launch")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--gpus", default="4,0,1")
    p.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    {"server": server, "worker": worker, "launch": launch}[args.command](args)


if __name__ == "__main__":
    main()
