import logging
import os
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Literal


def parse_arguments(argv: list[str]) -> Namespace:
    """Parses command-line arguments"""
    argument_parser = ArgumentParser(description="Calculate HDL")

    argument_parser.add_argument("--datastore-database-uri", type=str, required=True)
    argument_parser.add_argument("--data-module", type=str, required=True)

    argument_parser.add_argument("--artifact-store-path", type=Path, required=True)

    argument_parser.add_argument("--epoch-count", type=int, default=50)
    argument_parser.add_argument("--batch-size", type=int, default=16)
    argument_parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    argument_parser.add_argument("--seed", type=int, default=0)

    argument_parser.add_argument("--debug", action="store_true", default=False)
    argument_parser.add_argument(
        "--log-level", choices=logging.getLevelNamesMapping().keys(), default="INFO"
    )
    argument_parser.add_argument("--num-threads", type=int, default=None)

    return argument_parser.parse_args(argv)


def train() -> None:
    run_train(sys.argv[1:])


def run_train(
    argv: list[str], error_action: Literal["raise", "ignore"] = "ignore"
) -> None:
    arguments = parse_arguments(argv)

    os.environ["C_INCLUDE_PATH"] = ":".join(
        c_flag.removeprefix("-I")
        for c_flag in os.environ["CFLAGS"].split()
        if c_flag.startswith("-I")
    )

    from ..logging import logger

    try:
        from ..data.schema import Datastore

        datastore = Datastore(
            database_uri=arguments.datastore_database_uri,
            cache_path=arguments.artifact_store_path,
        )

        from ..data import datamodule
        from .train import Trainer

        data_module_class = getattr(datamodule, arguments.data_module)

        trainer = Trainer(
            datastore=datastore,
            data_module_class=data_module_class,
            train_batch_size=arguments.batch_size,
            gradient_accumulation_steps=arguments.gradient_accumulation_steps,
            eval_batch_size=arguments.batch_size,
            seed=arguments.seed,
            epoch_count=arguments.epoch_count,
        )

        trainer.objective()
    except Exception as e:
        logger.exception("Exception: %s", e, exc_info=True)
        if arguments.debug:
            import pdb

            pdb.post_mortem()
        if error_action == "raise":
            raise e


if __name__ == "__main__":
    train()
