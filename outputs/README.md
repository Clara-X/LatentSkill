# Archived experiment results

These completed runs are explicitly allowed by `.gitignore`. Preserve the raw
reports and traces alongside the code; logs, progress files, process IDs, caches,
aborted runs, and newly created run directories remain ignored.

| Directory | Evidence | Interpretation |
| --- | --- | --- |
| `boundary_gate/` | Initial paired-state probe, labels, audits, and gate artifacts | Local next-action probe; not full-task success |
| `alfworld_stage1_paired/` | 30 seen development games, three conditions, serial checks, and probe/budget audits | Batched inference; 512-token cap is a major confound |
| `alfworld_unseen_4096_20261022_173236/` | 134 unseen games, 402 native evaluation episodes | Real positive/negative transfer; already inspected evaluation data |
| `alfworld_stage2a_train_v3/` | Train-only scene-separated manifests, overlap audit, 179 real counterfactual states, gate models and validation predictions | Fixed-state continuation comparison; sealed test remains unexecuted |
| `reproduce_searchqa_full/` | SearchQA result traces, metrics, analysis, and launch script | Separate benchmark reproduction; see its analysis report |

See [Stage 1](../evals/boundary_gate/STAGE1_RESULTS.md),
[unseen analysis](../evals/boundary_gate/UNSEEN_4096_RESULTS.md), and
[Stage 2A results](../evals/boundary_gate/STAGE2A_RESULTS.md) for conclusions and
limitations. `plan.json`/manifests and audit hashes record the original settings.
Historical absolute paths describe the original server: rebase paths in a working
copy when replaying on another machine, rather than overwriting archived evidence.

Model checkpoints and source datasets are backed up separately. Gate pickle files
are small saved experiment artifacts; load only trusted local files. Adding a new
run to this archive requires an explicit `.gitignore` allowlist entry, including a
JSONL exception when its raw trajectories are needed.
