import pdb
import sys
import traceback
from argparse import ArgumentParser, Namespace
from contextlib import nullcontext
from functools import cache, partial
from multiprocessing import get_context, parent_process
from pathlib import Path
from subprocess import check_output
from typing import (
    Any,
    Callable,
    ContextManager,
    Iterable,
    Iterator,
    Mapping,
    NamedTuple,
    TypeVar,
)

from tqdm.auto import tqdm

from file_index.bids import BIDSIndex

from ..data.compression import compress_image
from ..data.schema import Datastore
from .base import Report
from .bold_conf import parse_bold_conf
from .norm import parse_norm
from .skull_strip import parse_skull_strip
from .tsnr import parse_tsnr

multiprocessing_context = get_context("forkserver")


@cache
def cpu_count() -> int:
    return int(check_output(["nproc"]).decode().strip())


T = TypeVar("T")
S = TypeVar("S")


def make_pool_or_null_context(
    iterable: Iterable[T],
    callable: Callable[[T], S],
    processes: int = 1,
    chunksize: int = 1,
) -> tuple[ContextManager[Any], Iterator[S]]:
    if processes < 2:
        return nullcontext(), map(callable, iterable)

    pool = multiprocessing_context.Pool(processes=processes)
    output_iterator: Iterator[S] = pool.imap_unordered(callable, iterable, chunksize)
    return pool, output_iterator


ImageParser = Callable[[Path], Iterator[Report]]
image_parsers: dict[str, ImageParser] = dict(
    skull_strip_report=parse_skull_strip,
    t1_norm_rpt=parse_norm,
    tsnr_rpt=parse_tsnr,
    bold_conf=parse_bold_conf,
    epi_norm_rpt=parse_norm,
)


class Job(NamedTuple):
    parse: ImageParser
    path: Path
    tags: Mapping[str, str]


class CompressedReport(NamedTuple):
    image_bytes: bytes
    direction: str | None
    i: int | None
    tags: Mapping[str, str]


def parse_image(job: Job, debug: bool = False) -> Iterable[CompressedReport]:
    parse, path, tags = job

    compressed_reports: list[CompressedReport] = list()
    try:
        for report in parse(path):
            direction, i, image = report
            image_bytes = compress_image(image)
            compressed_reports.append(CompressedReport(image_bytes, direction, i, tags))
    except ValueError:
        print(f"Error parsing {path}: {traceback.format_exc()}", flush=True)
        if debug and parent_process() is None:
            pdb.post_mortem()

    return compressed_reports


def main() -> None:
    run(sys.argv[1:])


def parse_arguments(argv: list[str]) -> Namespace:
    argument_parser = ArgumentParser()

    argument_parser.add_argument("--database", required=True)
    argument_parser.add_argument("--dataset", required=True)
    argument_parser.add_argument("--suffix", required=True)

    argument_parser.add_argument("--debug", action="store_true", default=False)
    return argument_parser.parse_args(argv)


def run(argv: list[str]) -> None:
    arguments = parse_arguments(argv)

    try:
        ingest(arguments)
    except Exception as e:
        if arguments.debug:
            import pdb

            pdb.post_mortem()
        else:
            raise e


def ingest(arguments: Namespace) -> None:
    database_path = Path(arguments.database)

    dataset_path = Path(arguments.dataset)
    index = BIDSIndex()
    index.put(dataset_path)
    for path in index.paths - index.get(extension=".svg"):
        index.remove(path)

    datastore = Datastore(database=str(database_path))

    suffix = arguments.suffix
    parse = image_parsers[suffix]

    paths = index.get(suffix=suffix)
    jobs = [Job(parse, path, index.get_tags(path)) for path in paths]
    total = len(jobs)

    processes = cpu_count()
    chunksize, extra = divmod(total, processes * 2**5)
    if extra:
        chunksize += 1
    pool, iterator = make_pool_or_null_context(
        jobs, partial(parse_image, debug=arguments.debug), processes, chunksize
    )
    with datastore, pool:
        datastore.set_tags_from_index(index)

        for compressed_reports in tqdm(
            iterator, total=total, unit="images", desc="parsing images"
        ):
            with datastore.connection:
                for compressed_report in compressed_reports:
                    datastore.add_image(*compressed_report)
