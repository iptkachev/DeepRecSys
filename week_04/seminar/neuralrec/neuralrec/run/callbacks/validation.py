from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from neuralrec.run.callbacks.base import Callback

if TYPE_CHECKING:
    from neuralrec.data.dataloader import DataLoader
    from neuralrec.run.train import TrainRunner


class ValidationCallback(Callback):
    def __init__(
        self,
        val_loader: DataLoader[object] | None = None,
    ) -> None:
        self._val_loader = val_loader

    def on_step_end(
        self,
        runner: TrainRunner,
        batch: object,
        out: dict[str, Any],
    ) -> None:
        if self._val_loader is None:
            return
        runner.model.eval()
        total_loss = 0.0
        n_batches = 0
        with torch.inference_mode():
            for batch in self._val_loader:
                batch_out = runner._model_forward(batch)
                total_loss += float(batch_out["loss"])
                n_batches += 1
        out["validation/loss"] = total_loss / max(n_batches, 1)
