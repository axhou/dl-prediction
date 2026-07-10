"""Stock-return prediction package."""

from .config import TrainingConfig, get_preset
from .experiment import ModelRunSpec, run_model_suite

__all__ = ["ModelRunSpec", "TrainingConfig", "get_preset", "run_model_suite"]
