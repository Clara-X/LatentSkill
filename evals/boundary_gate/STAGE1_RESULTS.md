# Stage 1 results: complete-task LoRA diagnostic

The 30-game development diagnostic is complete: 90 full episodes, plus six
serial verification episodes. The frozen LoRA improves success under the native
512-token generation budget, but heavy truncation prevents attributing the gain
solely to learned execution knowledge. No new gate was trained in this stage.

## Protocol

Use the `latentskill` conda environment. The predeclared `valid_seen` manifest
contains ten heat, ten cool, and ten clean games, seed 42. Scenes 6, 11, and 21
remain reserved. Each condition uses the original ALFWorld evaluator, prompts,
chat template, five-step history, greedy thinking generation, tag-based action
parser, and maximum 50 environment steps. Checkpoint:
`checkpoints/latentskill_sft_qwen3_8b/checkpoint-epoch-10`; skill texts:
`evals/alfworld/skills`. This is development evidence, not final held-out testing.

## Complete-task outcomes

| Condition | Successes | Success rate |
| --- | ---: | ---: |
| Base | 1/30 | 3.3% |
| Always On | 7/30 | 23.3% |
| Textual Skill | 4/30 | 13.3% |
| Retrospective Base/On oracle | 8/30 | 26.7% |
| Retrospective three-condition oracle | 10/30 | 33.3% |

Always On yields seven beneficial, one harmful, and 22 equal-return games versus
Base: a net improvement of six successes (20 percentage points). The oracle
selects whole completed branches after observing returns; it is not a deployed
state gate. Base/On oracle headroom above Always On is one success in this sample.

| Skill | Base | Always On | Textual Skill | Beneficial / harmful / neutral On games |
| --- | ---: | ---: | ---: | --- |
| Heat | 0/10 | 1/10 | 1/10 | 1 / 0 / 9 |
| Cool | 1/10 | 1/10 | 0/10 | 1 / 1 / 8 |
| Clean | 0/10 | 5/10 | 3/10 | 5 / 0 / 5 |

## Audits and limits

- All 30 initial Base/On prompt hashes match. Adapters are frozen, nonzero,
  skill-specific, and change logits and generated outputs. Zero-adapter and
  cached-linear-weight controls have zero logit difference on audited inputs.
- Generation hits the 512-token cap on 82.4% of Base steps, 24.2% of On steps,
  and 86.0% of Textual Skill steps. Parse failures are 82.3%, 27.0%, and 85.9%,
  respectively. These are pooled step rates, not task failure rates.
- On three initial development states, raising the cap to 2048 restores Base
  action tags from 1/3 to 3/3. This is a single-state audit, not a full-task
  higher-budget comparison. LoRA gains may include shorter reasoning and better
  formatting; knowledge gains have not been isolated.
- GPU batching can change long greedy outputs relative to serial inference.
  Matched Base/On rows share a batch, but this does not establish serial
  equivalence. Three preselected serial games agree with batched success labels;
  both conditions fail all three, so this is weak evidence about robustness.
- On six old probe states, mean-log-probability choices agree with free
  generation under the same probe prefix in 12/12 mode-state checks; reversing
  candidates changes one choice. That prefix forces empty thinking and two
  choices, unlike full-task evaluation. Its labels cannot replace task returns.

## Artifacts and next decision

Artifacts are under `outputs/alfworld_stage1_paired/`: manifests, all episode
traces, `full_task_report.json`, `first_divergence_states.jsonl`, probe/budget
audits, service controls, and serial verification. The 30 divergence labels
compare remaining-policy episode returns, not unique correct next actions.
See [protocol and commands](FULL_TASK_DIAGNOSTICS.md) for reproduction.

The fixed-budget execution diagnostic supports continuing the research, including
real negative transfer, but does not establish generalizable skill boundaries.
Before treating these labels as gate-training targets, compare complete Base/On
rollouts at a larger generation budget and expand harmful examples grouped by
game and scene. Keep reserved test scenes sealed.
