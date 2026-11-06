# Initial boundary pilot: 2026-10-18

## Setup

- Environment: `latentskill` conda environment, NVIDIA A100 80 GB, physical GPU 6.
- Frozen checkpoint: `checkpoints/latentskill_sft_qwen3_8b/checkpoint-epoch-10`.
- Skills: existing `evals/alfworld/skills/{heat,cool,clean}.txt`.
- Seed: 42. Per skill: 4 training, 2 validation, 2 test games.
- Total: 24 real ALFWorld state pairs, or 48 states (24 train / 12 val / 12 test).
- The test games use unseen scene 10; this is **one held-out room**, not broad
  scene coverage. Task families and paired states never cross splits.
- Inference: float32 stored model weights with bfloat16 autocast; mean-token
  log probability over two candidate next actions. See [protocol](README.md).

## Held-out results

Accuracy below means correct next action in this constrained probe, not episode
success. Negative transfer counts previously correct base decisions made incorrect.

| Policy | Correct next action | Negative transfer vs base | Activation rate |
|---|---:|---:|---:|
| Always off | 6/12 (50.0%) | 0/12 | 0% |
| Always on | 6/12 (50.0%) | 1/12 | 100% |
| Operation-applicability gate | 5/12 (41.7%) | 1/12 | 50% |
| Observed-benefit gate | 6/12 (50.0%) | 0/12 | 0% |
| Retrospective oracle | 7/12 (58.3%) | 0/12 | 8.3% |

The applicability gate classified all 12 operation-precondition labels correctly.
However, this did not identify which states benefited from the workflow adapter:

- **Harm while applicable:** holding lettuce at the fridge, the base selected
  `cool lettuce 2 with fridge 1`. LoRA instead selected `go to fridge 1`, which
  was inadmissible and produced `Nothing happens.`
- **Benefit while the operation was inapplicable:** away from the microwave,
  the heat adapter correctly selected navigation, while the base attempted to
  heat the object remotely. The full workflow adapter can help before its target
  operation is executable.

There was only **one beneficial training state**, no harmful training states,
and **no on/off reward differences in validation**. Validation therefore selected
the benefit gate's never-activate threshold. Its result equals the always-off
baseline; it does not demonstrate learned selective reuse.

## Interpretation and artifacts

This pilot verifies the collection, frozen-adapter comparison, gate training,
and state-level selection pipeline. It **does not establish that gating improves
performance**. Operation validity and adapter benefit are different labels; the
next dataset needs multiple beneficial and harmful states in training and
validation, additional boundary types, counterbalanced or free-generation
evaluation, more scenes, and full rollouts. Keep this test set held out while
designing those extensions.

Local artifacts are in `outputs/boundary_gate/`: `pairs.jsonl`, `outcomes.jsonl`,
their manifests, and `trained/{report.json,test_predictions.jsonl,*_gate.pkl}`.
The report records split counts, input hash, package versions, thresholds, and
per-skill metrics. Four unit tests passed; real simulator replay and CUDA inference
completed successfully. Generated datasets, models, and logs remain untracked.
