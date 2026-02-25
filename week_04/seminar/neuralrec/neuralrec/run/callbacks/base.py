from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from neuralrec.run.train import TrainRunner


class Callback(ABC):
    def on_train_begin(self, runner: TrainRunner) -> None:
        pass

    def on_train_end(self, runner: TrainRunner) -> None:
        pass

    def on_epoch_end(self, runner: TrainRunner) -> None:
        pass

    def on_step_begin(self, runner: TrainRunner, batch: Any) -> None:
        pass

    def on_step_end(
        self,
        runner: TrainRunner,
        batch: Any,
        out: dict[str, Any],
    ) -> None:
        pass

    def on_before_optimizer_step(self, runner: TrainRunner) -> None:
        pass

    def every_n_steps(
        self,
        n: int,
        include_step_zero: bool = False,
    ) -> Callback:
        callback = self

        class EveryNStepsCallback(Callback):
            def _run_if_step(self, runner: TrainRunner, fn: Callable[..., None], *args: Any) -> None:
                step = runner.step
                if (step == 0 and include_step_zero) or (step != 0 and step % n == 0):
                    fn(*args)

            def on_train_begin(self, runner: TrainRunner) -> None:
                callback.on_train_begin(runner)

            def on_train_end(self, runner: TrainRunner) -> None:
                callback.on_train_end(runner)

            def on_epoch_end(self, runner: TrainRunner) -> None:
                callback.on_epoch_end(runner)

            def on_step_begin(self, runner: TrainRunner, batch: Any) -> None:
                self._run_if_step(runner, callback.on_step_begin, runner, batch)

            def on_step_end(
                self,
                runner: TrainRunner,
                batch: Any,
                out: dict[str, Any],
            ) -> None:
                self._run_if_step(runner, callback.on_step_end, runner, batch, out)

            def on_before_optimizer_step(self, runner: TrainRunner) -> None:
                self._run_if_step(runner, callback.on_before_optimizer_step, runner)

        return EveryNStepsCallback()

    def ignore_if(self, condition: bool) -> Callback:
        callback = self

        class IgnoreIfCallback(Callback):
            def _run_unless_ignored(self, runner: TrainRunner, fn: Callable[..., None], *args: Any) -> None:
                if not condition:
                    fn(*args)

            def on_train_begin(self, runner: TrainRunner) -> None:
                self._run_unless_ignored(runner, callback.on_train_begin, runner)

            def on_train_end(self, runner: TrainRunner) -> None:
                self._run_unless_ignored(runner, callback.on_train_end, runner)

            def on_epoch_end(self, runner: TrainRunner) -> None:
                self._run_unless_ignored(runner, callback.on_epoch_end, runner)

            def on_step_begin(self, runner: TrainRunner, batch: Any) -> None:
                self._run_unless_ignored(runner, callback.on_step_begin, runner, batch)

            def on_step_end(
                self,
                runner: TrainRunner,
                batch: Any,
                out: dict[str, Any],
            ) -> None:
                self._run_unless_ignored(runner, callback.on_step_end, runner, batch, out)

            def on_before_optimizer_step(self, runner: TrainRunner) -> None:
                self._run_unless_ignored(runner, callback.on_before_optimizer_step, runner)

        return IgnoreIfCallback()