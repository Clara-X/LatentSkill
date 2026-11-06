# Stage 1: frozen LoRA execution diagnostics

## README configuration: full unseen follow-up

The follow-up requested on 2026-10-22 uses the README settings: full `unseen`,
50 environment steps, 4096 generated tokens, and both tokenized context limits
4096. Run it with the `latentskill` environment:

```bash
python -u -m evals.boundary_gate.native_eval launch \
  --output-dir outputs/alfworld_unseen_4096_NEW_RUN \
  --gpus 0,1,2,3,4,5,6,7
```

Choose a new output directory. The launcher obtains all 134 eligible unseen
games from the configured ALFWorld environment and partitions them into eight
disjoint manifests. Each GPU runs Base, Always On, and Textual Skill sequentially
on its own manifest: 402 complete episode evaluations in total. Every subprocess
calls `python -m evals.alfworld.evaluate` directly, without the earlier batching
service, autocast wrapper, or cached weight conversion. The only additions to the
README arguments are the baseline selector, shard manifest, seed 42, and prompt
recording. GPU concurrency changes neither the selected task set nor decoding.

`plan.json` stores exact arguments and skill hashes. Per-mode/shard directories
contain `job.json` (PID and exit status), `alfworld.log`, `progress.json`, and the
official result files. On successful completion, the launcher validates complete,
duplicate-free game coverage and matching initial Base/On prompt hashes before
writing `full_task_report.json`. Rebuild the report with:

```bash
python -m evals.boundary_gate.native_eval report \
  --output-dir outputs/alfworld_unseen_4096_NEW_RUN
```

This follow-up changes both split and token budget relative to the 30-game seen
diagnostic below. Differences between those runs cannot isolate budget effects.
Unseen results are evaluation evidence and must not become gate-training data;
some unseen states were inspected in the earlier pilot, so the complete split
is not an untouched final test set for the research project.

## Original 30-game development diagnostic

This experiment deliberately pauses gate training. It compares **Base**, **Always
On**, and **Textual Skill** on the same 30 complete development tasks: ten each
for heat, cool, and clean. Episode-level Base/On and three-condition retrospective
oracles select successful branches only after execution; they are not deployed.

## Protocol and data

The runner calls `evals.alfworld.evaluate.main`. Original prompt construction,
five-step history, skill selection, chat template, greedy thinking generation,
action parser, environment stepping, and 50-step budget remain in that entrypoint.
The generation budget is its default 512 tokens. The entrypoint initially loads
float32 parameters and all conditions use bfloat16 autocast. Linear weights cache the exact bf16 cast
that autocast would otherwise repeat on every token; embeddings and normalization
remain float32. A runtime check requires this cache to leave both Base/On logits
bit-identical on an audited development prompt. Seed: 42. No weights are trained.

`diagnose setup` samples solvable `valid_seen` task families before evaluation,
with no duplicate families. Scenes 6, 11, and 21 are reserved for future gate
testing and never used in this stage. This is a **development diagnostic**, not
an independent final test or a claim about the backbone's training-data overlap.

The only evaluator additions are optional manifest filtering, explicit seeding,
prompt/token metadata, and progress recording. Defaults remain unchanged.

## Run with the latentskill environment

```bash
conda activate latentskill
python -m evals.boundary_gate.diagnose setup \
  --output-dir outputs/alfworld_stage1_paired/manifests
python -u -m evals.boundary_gate.batched_diagnose launch \
  --output-dir outputs/alfworld_stage1_paired --gpus 4,0,1 --batch-size 16
python -m evals.boundary_gate.diagnose audit-probe \
  --output outputs/alfworld_stage1_paired/probe_audit.jsonl
python -m evals.boundary_gate.diagnose audit-budget \
  --serial-root outputs/alfworld_stage1 \
  --output outputs/alfworld_stage1_paired/budget_audit.jsonl --budget 2048
python -m evals.boundary_gate.diagnose report \
  --output-dir outputs/alfworld_stage1_paired
python -m unittest discover -s tests -v
```

Use a fresh output directory. The existing single-process entrypoint can also be
used directly, e.g.:

```bash
CUDA_VISIBLE_DEVICES=4 python -u -m evals.boundary_gate.diagnose run \
  --mode on --manifest outputs/alfworld_stage1_paired/manifests/heat.json \
  --output-dir outputs/alfworld_stage1_serial_check/on/heat
```

## Batched execution and controls

CPU workers execute the original evaluator for each game. A local Unix-socket
service batches only model-generation requests, preserving input IDs and decoding
settings. It synchronizes rounds and keeps a game's Base/On rows in the same
kernel batch, including while their prompts are identical. Different games and
Textual Skill requests cannot split a matched pair. Base/Text rows use zero
adapter tensors; a runtime check requires their logits to exactly match a true
`adapter_state=None` control. Adapter tensors remain frozen and are hashed.

Batch padding and low-precision kernels can change long greedy reasoning relative
to serial execution. Serial audit output and two-row batch output are retained
when available; results are **not asserted to be bit-identical serial replications**.
Within-run matched pairing controls this source of Base/On differences. GPU
service metadata records precision, adapter hashes, scale, and zero-control error.
The optional budget audit uses three already inspected development prompts at
2048 tokens, without changing the primary experiment or claiming episode success
from a single action.

## Measurements and decision rule

`full_task_report.json` reports simulator task success, positive/negative/neutral
transfer, retrospective oracle potential, token truncation, parse failures,
context-length caps, and legally formed bare actions rejected by the original
tag-based parser. All three conditions must cover the predeclared game set.

`first_divergence_states.jsonl` retains the first distinct Base/On actions after an
identical tokenized prefix. Its gain label compares full remaining episode returns
under the same total budget, not agreement with a unique expert action. Other
reasonable strategies receive credit whenever the simulator task succeeds.

`audit-probe` separately checks the old six-state probe using mean/summed log
probabilities, reversed candidate ordering, free generation with exactly the same
probe prefix, and simulator command validity. These previously inspected states
are diagnostics, not new held-out evidence. The old probe forced empty thinking
and a two-choice prompt; it does not match the official thinking/history protocol.

If full-task LoRA gains exist but the probe misses them, replace the local probe
with trajectory-return labels. If gains are absent, distinguish genuine skill
weakness from truncation, parser, context, loading, or scaling failures before
training a gate. A high truncation rate prevents attributing failures solely to
the skill representation. Keep subsequent test scenes sealed during diagnosis.
