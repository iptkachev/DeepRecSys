from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from neuralrec.run.callbacks.base import Callback

if TYPE_CHECKING:
    from neuralrec.data.dataloader import DataLoader


class TrainRunner:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader[Any],
        callbacks: list[Callback] | None = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.callbacks = list(callbacks or [])

        self.current_epoch = 0
        self.step = 0  # step within current epoch (0-based, reset each epoch)
        self.global_step = 0  # total steps across all epochs

    def _model_forward(self, batch: Any) -> dict[str, Any]:
        out = self.model(batch)
        if not isinstance(out, dict) or "loss" not in out:
            raise ValueError("Model must return a dict with 'loss' key")
        return out

    def train_epoch(self, epoch: int) -> None:
        self.current_epoch = epoch
        self.step = 0
        self.model.train()
        for batch in self.train_loader:
            for cb in self.callbacks:
                cb.on_step_begin(self, batch)
            self.optimizer.zero_grad(set_to_none=True)
            out = self._model_forward(batch)
            loss = out["loss"]
            loss.backward()
            for cb in self.callbacks:
                cb.on_step_end(self, batch, out)
            for cb in self.callbacks:
                cb.on_before_optimizer_step(self)
            self.optimizer.step()
            self.step += 1
            self.global_step += 1
        for cb in self.callbacks:
            cb.on_epoch_end(self)

    def fit(self, num_epochs: int) -> None:
        for cb in self.callbacks:
            cb.on_train_begin(self)
        for epoch in range(num_epochs):
            self.train_epoch(epoch)
        for cb in self.callbacks:
            cb.on_train_end(self)
