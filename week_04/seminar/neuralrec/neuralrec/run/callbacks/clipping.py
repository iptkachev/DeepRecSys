from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from neuralrec.run.callbacks.base import Callback

if TYPE_CHECKING:
    from neuralrec.run.train import TrainRunner


class GradientNormClippingCallback(Callback):
    def __init__(self, max_norm: float) -> None:
        self._max_norm = max_norm

    def on_before_optimizer_step(self, runner: TrainRunner) -> None:
        model = getattr(runner, "model", None)
        if model is None:
            return
        torch.nn.utils.clip_grad_norm_(model.parameters(), self._max_norm)
