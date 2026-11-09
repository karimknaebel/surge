from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import timedelta

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    ProgressColumn,
    Task,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.text import Text


def _duration(seconds: float | None) -> str:
    return (
        str(timedelta(seconds=max(0, int(seconds)))) if seconds is not None else "--:--"
    )


class _SpeedColumn(ProgressColumn):
    def render(self, task: Task) -> Text:
        if task.fields["state"] != "running":
            return Text()
        speed = task.speed
        return Text(f"{speed:.2f} img/s" if speed is not None else "-- img/s")


class _RemainingColumn(TimeRemainingColumn):
    def render(self, task: Task) -> Text:
        if task.fields["state"] != "running":
            return Text()
        return Text.assemble("ETA ", super().render(task))


class EvaluationProgress(Progress):
    """Keep dataset rows in terminals and emit occasional plain lines in logs."""

    def __init__(
        self,
        dataset_count: int,
        *,
        console: Console | None = None,
        get_time: Callable[[], float] | None = None,
        log_interval: float = 30,
    ):
        console = console or Console(stderr=True)
        super().__init__(
            TextColumn("{task.description}", markup=False),
            BarColumn(bar_width=None),
            MofNCompleteColumn(),
            TextColumn("{task.fields[state]}"),
            TextColumn("elapsed"),
            TimeElapsedColumn(),
            _SpeedColumn(),
            _RemainingColumn(),
            console=console,
            get_time=get_time,
            refresh_per_second=4,
            disable=not console.is_interactive,
        )
        self.overall = self.add_task("Datasets", total=dataset_count, state="complete")
        self.log_interval = log_interval
        self._last_log = self.get_time()

    def _print(self, message: str) -> None:
        self.console.print(message, markup=False, highlight=False, soft_wrap=True)

    def _log_dataset(self, task: Task) -> None:
        total = int(task.total) if task.total is not None else "?"
        elapsed = task.finished_time if task.finished else task.elapsed
        message = f"{task.description}: {int(task.completed)}/{total} {task.fields['state']}; elapsed {_duration(elapsed)}"
        speed = task.speed
        if task.fields["state"] == "running" and speed is not None:
            message += f"; {speed:.2f} img/s; ETA {_duration(task.time_remaining)}"
        self._print(message)

    @contextmanager
    def dataset(self, name: str) -> Iterator[Task]:
        task_id = self.add_task(name, total=None, start=False, state="loading")
        task = next(task for task in self.tasks if task.id == task_id)
        self._last_log = self.get_time()
        if self.disable:
            self._print(f"{name}: starting")
        try:
            yield task
        except BaseException as error:
            self.update(
                task_id,
                state="interrupted"
                if isinstance(error, KeyboardInterrupt)
                else "failed",
            )
            raise
        else:
            self.update(task_id, state="done")
            self.advance(self.overall)
        finally:
            self.stop_task(task_id)
            if self.disable:
                self._log_dataset(task)

    def start_samples(self, task: Task, total: int) -> None:
        self.start_task(task.id)
        self.update(task.id, total=total, state="running")
        self._last_log = self.get_time()

    def advance_sample(self, task: Task) -> None:
        self.advance(task.id)
        now = self.get_time()
        if (
            self.disable
            and task.completed < task.total
            and now - self._last_log >= self.log_interval
        ):
            self._log_dataset(task)
            self._last_log = now

    def stop(self) -> None:
        self.stop_task(self.overall)
        super().stop()
        if self.disable:
            overall = next(task for task in self.tasks if task.id == self.overall)
            self._print(
                f"Datasets: {int(overall.completed)}/{int(overall.total)} complete; elapsed {_duration(overall.elapsed)}"
            )
