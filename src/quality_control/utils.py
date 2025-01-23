import json
import os
import signal
from contextlib import AbstractContextManager
from dataclasses import dataclass
from functools import cache
from hashlib import sha1
from subprocess import check_output
from time import time
from types import TracebackType
from typing import Any, Self, Sequence

import torch

from .logging import logger


@cache
def cpu_count() -> int:
    return int(check_output(["nproc"]).decode().strip())


def hex_digest(value: Any) -> str:
    hash = sha1()
    str_representation = json.dumps(value, sort_keys=True)
    hash.update(str_representation.encode())
    digest = hash.hexdigest()
    return digest


@dataclass
class Timer:
    start_time: float = float("nan")

    numerator: float = 0.0
    denominator: int = 0

    def start(self) -> None:
        self.start_time = time()

    def stop(self) -> None:
        self.numerator += time() - self.start_time
        self.denominator += 1

    @property
    def value(self) -> float:
        if self.denominator > 0:
            return self.numerator / self.denominator
        else:
            return float("nan")


class TrainingState(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("epoch_index", torch.tensor(0))
        self.register_buffer("step_index", torch.tensor(0))


num_threads_variables: Sequence[str] = [
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMEXPR_MAX_THREADS",
    "NPROC",
    "OPENCV_FFMPEG_THREADS",
]


def apply_num_threads(num_threads: int | None) -> None:
    for variable in num_threads_variables:
        os.environ[variable] = str(num_threads)


class Timeout(AbstractContextManager["Timeout"]):
    def __init__(self, seconds: int) -> None:
        self.seconds = int(seconds)
        signal.signal(signal.SIGALRM, self.handler)

    def handler(self, *args: Any) -> None:
        raise TimeoutError()

    def reset(self, seconds: int | None = None) -> None:
        if seconds is None:
            seconds = self.seconds
        signal.alarm(seconds)
        logger.info(f"Timeout set to {seconds} seconds")

    def __enter__(self) -> Self:
        self.reset()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        signal.alarm(0)
