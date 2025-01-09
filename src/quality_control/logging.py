import logging

import optuna as optuna
from optuna.logging import _default_handler

logger = logging.getLogger("quality_control")
logger.setLevel(logging.INFO)
if _default_handler is None:
    raise RuntimeError
logger.addHandler(_default_handler)
