# SPDX-License-Identifier: Apache-2.0
# Based on and adapted from SHINE: [https://github.com/MuLabPKU/SHINE](https://github.com/MuLabPKU/SHINE)
# Modified by the LatentSkill authors in 2026.

def freeze_backbone_except_memory(backbone):
    """Freeze backbone parameters while keeping memory tokens trainable."""
    for param in backbone.parameters():
        param.requires_grad = False
    backbone.model.mem_tokens.requires_grad = True
