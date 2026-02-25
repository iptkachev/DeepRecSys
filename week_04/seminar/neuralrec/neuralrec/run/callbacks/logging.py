from __future__ import annotations

from typing import TYPE_CHECKING, Any

from loguru import logger

from neuralrec.run.callbacks.base import Callback

if TYPE_CHECKING:
    from neuralrec.run.train import TrainRunner


class LoggingCallback(Callback):
    def __init__(self, level: str = "INFO") -> None:
        self._level = level.upper()

    def on_step_end(
        self,
        runner: TrainRunner,
        batch: Any,
        out: dict[str, Any],
    ) -> None:
        parts = [f"step={runner.step}", f"epoch={runner.current_epoch}"]
        for key, value in out.items():
            if hasattr(value, "detach"):
                try:
                    v = float(value.detach())
                    parts.append(f"{key}={v:.4f}")
                except (ValueError, TypeError):
                    parts.append(f"{key}={value.detach()}")
            else:
                parts.append(f"{key}={value}")
        logger.log(self._level, " ".join(parts))

    def on_epoch_end(self, runner: TrainRunner) -> None:
        logger.log(self._level, "epoch {} finished", runner.current_epoch)
