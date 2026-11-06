# Repository Guidelines

## Project Structure & Module Organization

LatentSkill compiles textual skills into LoRA adapters using a hypernetwork and a frozen Qwen3 backbone.

- `latentskill/models/`: hypernetwork, Qwen3 wrapper, and adapter operations.
- `latentskill/data/`: dataset loaders, collators, and group indexing.
- `latentskill/training/`: two-stage training and checkpoint handling; `latentskill/utils/` contains shared helpers.
- `configs/models/qwen3_8b.yaml`: Hydra configuration; `scripts/train/` contains launchers.
- `evals/alfworld/` and `evals/searchqa/`: benchmark runners, prompts, and skill-text variants.
- `data/`, `models/`, `checkpoints/`, `wiki_index/`, and `alfworld_data/`: external resources, generally represented by tracked placeholders. `huggingface/` documents release artifacts; `framework.png` illustrates the architecture.

## Build, Test, and Development Commands

Use Python 3.10, install PyTorch for your CUDA platform, then run `pip install -r requirements.txt`. Run commands from the repository root; there is no separate build step.

- `bash scripts/train/pretrain_qwen3_8b.sh`: train the compiler on skill documents.
- `bash scripts/train/sft_qwen3_8b.sh`: fine-tune using trajectories and a pretrained compiler checkpoint.
- `bash evals/alfworld/run_eval.sh <checkpoint-dir> unseen 0`: launch ALFWorld evaluation in the background; inspect its log for completion.
- `bash evals/searchqa/run_eval.sh <checkpoint-dir> 8000`: start or reuse retrieval and run SearchQA evaluation. Requires the wiki index, corpus, and E5 retriever.

Configure paths with `MODEL_PATH`, `DATA_ROOT`, and `CHECKPOINT_ROOT`. Training launchers accept overrides such as `NUM_GPUS` and `TRAIN_BATCH_SIZE`; check resource availability before launching.

## Coding Style & Naming Conventions

Follow existing Python style: four-space indentation, `snake_case` functions/modules, and `PascalCase` classes. Use two-space YAML indentation. Keep model settings in Hydra configuration and launcher overrides in environment variables. Preserve SPDX and attribution headers. No formatter or linter configuration is currently included.

## Testing Guidelines

No dedicated unit-test suite, testing framework, or coverage threshold is configured. Validate relevant changes with focused smoke runs and the affected benchmark. Record checkpoint, seed, split, skill directory, commands, and before/after metrics. If adding unit tests, use `tests/test_<module>.py` and document their runner and dependencies. Report checks skipped because models, data, or GPUs were unavailable.

## Commit & Pull Request Guidelines

Git history contains only `init`, so no established message convention exists. Use concise, imperative commit subjects. PRs should explain the change, list configuration or resource requirements, and report validation commands and results. Link relevant issues and flag checkpoint-format changes. Keep generated weights, datasets, logs, credentials, and local environment files out of commits.
