from dataclasses import dataclass
import json
from functools import cache
from hashlib import sha1
from subprocess import check_output
from time import time
from typing import Any


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
