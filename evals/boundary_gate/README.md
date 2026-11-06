# Frozen LoRA boundary pilot

See [initial results and limitations](RESULTS.md) for the first completed run.
The subsequent [Stage 1 complete-task results](STAGE1_RESULTS.md) compare Base,
Always On, and Textual Skill on 30 development games, with truncation and backend
limitations documented separately from the local probe.
The [full unseen 4096-token follow-up](UNSEEN_4096_RESULTS.md) reports 402 native
episodes, substantial positive and negative transfer, and transform-action failures.
See [Stage 2A](STAGE2A.md) for Train-only, scene-isolated, replay-verified
counterfactual data collection and the three-gate comparison.
The [Stage 2A results](STAGE2A_RESULTS.md) report that Benefit Gate avoids observed
negative transfer but retains fewer beneficial states than the task-type gate.

This experiment uses **real ALFWorld snapshots**, three existing skill adapters
(`heat`, `cool`, `clean`), and a small independently trained gate. It updates no
backbone, compiler, memory-token, or LoRA parameters.

## What this pilot measures

The handcoded expert acquires the target object and navigates to the appropriate
appliance. The positive state permits the transformation operation. The paired
negative state is obtained by one `go to` command to another receptacle: inventory,
goal, and object properties remain unchanged, but the location prerequisite fails.
Both branches are reset and replayed; the simulator checks command availability.

Two identical candidate actions are compared at both states: transform the object,
or navigate to the appliance. The correct next action switches across the pair.
The frozen model ranks candidates by mean token log probability, with LoRA enabled
and disabled under identical inference settings. The selected action is executed
in a fresh replay of the real simulator. Reward is correct-next-action agreement,
**not complete-task success**. This is a constrained local probe, not an ALFWorld
benchmark result or a measurement of unconstrained generation quality.

Operation applicability is a **proxy**, not ground truth that the full workflow
LoRA is harmful: an adapter may also help navigation. We therefore train both an
applicability gate and a benefit gate whose positive labels require measured
`reward_on > reward_off`; ties prefer the base model. Constant-label training
sets use an explicit constant head rather than fabricating preference examples.

## Run

Use the `latentskill` conda environment from the repository root:

```bash
conda activate latentskill
python -m evals.boundary_gate.prepare \
  --output outputs/boundary_gate/pairs.jsonl
CUDA_VISIBLE_DEVICES=6 python -u -m evals.boundary_gate.collect \
  --pairs outputs/boundary_gate/pairs.jsonl \
  --checkpoint checkpoints/latentskill_sft_qwen3_8b/checkpoint-epoch-10 \
  --output outputs/boundary_gate/outcomes.jsonl
python -m evals.boundary_gate.train \
  --input outputs/boundary_gate/outcomes.jsonl \
  --output-dir outputs/boundary_gate/trained
python -m unittest discover -s tests -v
```

Default sampling is 4 training, 2 validation, and 2 test games **per skill**:
48 states in 24 pairs. Sources are ALFWorld train, valid_seen, and valid_unseen,
respectively. Entire task families, including alternate trials, are isolated
across splits. The seed is 42. Collection requires CUDA, the Qwen3 backbone, an
unpacked compiler checkpoint, and ALFWorld data. It exclusively creates the
outcomes file; choose a new output name for a new run.

To check only boundary learning, pass `pairs.jsonl` directly to `train`. Such a run
does not measure LoRA utility. Gates use per-skill TF-IDF and logistic regression
on task, observation, inventory, and recent actions. Oracle actions, IDs, labels,
rewards, and simulator command-availability labels never enter gate features.
Thresholds are selected on validation only; test states remain held out.

## Outputs and interpretation

`report.json` compares always off, always on, both learned gates, and a retrospective
oracle. It reports boundary false/missed activations, next-action accuracy,
negative transfer, and the number of beneficial, harmful, and tied states, with
per-skill breakdowns. Dataset and collection manifests record seeds, arguments,
checkpoint file metadata, skill hashes, precision, and inference settings.
`test_predictions.jsonl` stores individual gate decisions.

```python
from latentskill.models.boundary_gate import BoundaryGate

gate = BoundaryGate.load("outputs/boundary_gate/trained/benefit_gate.pkl")
active_adapter = gate.select_adapter(current_state, frozen_skill_adapter)
# backbone.generate(..., adapter_state=active_adapter, ignore_mem_token=True)
```

Load only trusted local pickle files. Small-sample improvements are exploratory;
if LoRA on/off never changes outcomes, this probe provides no evidence for learning
useful activation boundaries. Extend to more states, boundary types, repeated
seeds, free generation, and full rollouts before making a generalization claim.
