# SPDX-License-Identifier: Apache-2.0
# Based on and adapted from SHINE: [https://github.com/MuLabPKU/SHINE](https://github.com/MuLabPKU/SHINE)
# Modified by the LatentSkill authors in 2026.

import math

import torch
from transformers import get_linear_schedule_with_warmup


def build_optimizer_and_scheduler(grouped_params, train_loader, cfg, device):
    optimizer = torch.optim.AdamW(grouped_params, lr=cfg.optim.learning_rate)

    total_steps = cfg.optim.num_epochs * math.ceil(len(train_loader) / max(1, cfg.run.gradient_accumulation_steps))
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.optim.warmup_steps,
        num_training_steps=total_steps,
    )

    return optimizer, lr_scheduler
