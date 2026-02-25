from __future__ import annotations

from typing import TYPE_CHECKING, Any

from neuralrec.run.callbacks.base import Callback

if TYPE_CHECKING:
    from neuralrec.run.train import TrainRunner


class TensorBoardCallback(Callback):
    def __init__(
        self,
        log_dir: str = "runs",
    ) -> None:
        self._log_dir = log_dir
        self._writer: Any | None = None
        self._step = 0

    def on_train_begin(self, runner: TrainRunner) -> None:
        try:
            from torch.utils.tensorboard import SummaryWriter
            self._writer = SummaryWriter(log_dir=self._log_dir)
        except Exception as e:
            raise RuntimeError("TensorBoard callback requires torch.utils.tensorboard") from e
        self._step = 0

    def on_train_end(self, runner: TrainRunner) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    def on_step_end(
        self,
        runner: TrainRunner,
        batch: Any,
        out: dict[str, Any],
    ) -> None:
        if self._writer is None:
            return
        for key, value in out.items():
            if hasattr(value, "detach"):
                try:
                    v = float(value.detach())
                except (ValueError, TypeError):
                    try:
                        v = float(value.detach().mean())
                    except (ValueError, TypeError):
                        continue
            elif isinstance(value, (int, float)):
                v = float(value)
            else:
                continue
            self._writer.add_scalar(f"{key}", v, runner.global_step)
        self._step += 1
