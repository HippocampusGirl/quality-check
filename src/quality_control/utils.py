from functools import cache
from subprocess import check_output


@cache
def cpu_count() -> int:
    return int(check_output(["nproc"]).decode().strip())
