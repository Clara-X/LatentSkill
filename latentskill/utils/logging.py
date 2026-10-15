# SPDX-License-Identifier: Apache-2.0
# Based on and adapted from SHINE: [https://github.com/MuLabPKU/SHINE](https://github.com/MuLabPKU/SHINE)
# Modified by the LatentSkill authors in 2026.

import logging


def get_logger(name="Default", filename=None):
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
        filename=filename,
    )
    return logging.getLogger(name)
