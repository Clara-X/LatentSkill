#!/usr/bin/env bash
set -euo pipefail
cd /mnt/yulan/xujiahui/projects/LatentSkill
export CUDA_VISIBLE_DEVICES=5 PYTHONPATH="$PWD" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
/mnt/yulan/cache/conda/envs/latentskill/bin/python -u -c 'import random,numpy as np,torch,runpy; random.seed(42); np.random.seed(42); torch.manual_seed(42); runpy.run_module("evals.searchqa.evaluate",run_name="__main__")' --checkpoint checkpoints/latentskill_sft_qwen3_8b/checkpoint-epoch-10 --test_data /mnt/yulan/xujiahui/projects/LatentSkill/data/search_test/search_test_all.jsonl --retrieval_url http://127.0.0.1:8031/retrieve --retrieval_topk 3 --max_steps 4 --max_new_tokens 2048 --context_max_length 4096 --conversation_max_length 4096 --output_dir /mnt/yulan/xujiahui/projects/LatentSkill/outputs/reproduce_searchqa_full/results --device cuda
