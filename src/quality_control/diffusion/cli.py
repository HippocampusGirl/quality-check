import logging
import os
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Literal

from tqdm.contrib.logging import logging_redirect_tqdm


def parse_arguments(argv: list[str]) -> Namespace:
    """Parses command-line arguments"""
    argument_parser = ArgumentParser(description="Calculate HDL")

    argument_parser.add_argument("--datastore-database-uri", type=str, required=True)
    argument_parser.add_argument("--data-module", type=str, required=True)

    argument_parser.add_argument("--autoencoder-path", type=Path, required=True)
    argument_parser.add_argument("--artifact-store-path", type=Path, required=True)

    argument_parser.add_argument("--optuna-database-uri", type=str, required=True)
    argument_parser.add_argument("--optuna-study-name", type=str, required=True)
    argument_parser.add_argument("--trial-count", type=int, default=50)

    argument_parser.add_argument("--epoch-count", type=int, default=50)
    argument_parser.add_argument("--batch-size", type=int, default=16)
    argument_parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    argument_parser.add_argument("--timestep-count", type=int, default=1000)
    argument_parser.add_argument("--seed", type=int, default=0)

    argument_parser.add_argument("--debug", action="store_true", default=False)
    argument_parser.add_argument("--profile", action="store_true", default=False)
    argument_parser.add_argument(
        "--log-level", choices=logging.getLevelNamesMapping().keys(), default="INFO"
    )
    argument_parser.add_argument("--num-threads", type=int, default=None)

    return argument_parser.parse_args(argv)


def train() -> None:
    with logging_redirect_tqdm():
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

        from optuna.artifacts import FileSystemArtifactStore

        optuna_artifact_store_path = arguments.artifact_store_path / "optuna"
        optuna_artifact_store_path.mkdir(parents=True, exist_ok=True)
        artifact_store = FileSystemArtifactStore(optuna_artifact_store_path)

        from ..data import datamodule
        from .train import Trainer

        data_module_class = getattr(datamodule, arguments.data_module)

        from ..autoencoder.model import model_class

        autoencoder_model = model_class.from_pretrained(
            arguments.autoencoder_path
        ).eval()

        trainer = Trainer(
            datastore=datastore,
            data_module_class=data_module_class,
            autoencoder_model=autoencoder_model,
            artifact_store=artifact_store,
            batch_size=arguments.batch_size,
            gradient_accumulation_steps=arguments.gradient_accumulation_steps,
            timestep_count=arguments.timestep_count,
            seed=arguments.seed,
            epoch_count=arguments.epoch_count,
            is_profile=arguments.profile,
        )

        import torch

        torch.multiprocessing.set_sharing_strategy("file_system")  # type: ignore

        import optuna
        from optuna.storages import RetryFailedTrialCallback

        sampler = optuna.samplers.TPESampler(seed=arguments.seed)
        pruner = optuna.pruners.HyperbandPruner()
        storage = optuna.storages.RDBStorage(
            arguments.optuna_database_uri,
            heartbeat_interval=1,
            grace_period=10,
            failed_trial_callback=RetryFailedTrialCallback(
                inherit_intermediate_values=True
            ),
        )
        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
            study_name=arguments.optuna_study_name,
            storage=storage,
            load_if_exists=True,
        )
        study.optimize(
            trainer.objective,
            n_trials=arguments.trial_count,
            gc_after_trial=True,
        )
    except Exception as e:
        logger.exception("Exception: %s", e, exc_info=True)
        if arguments.debug:
            import pdb

            pdb.post_mortem()
        if error_action == "raise":
            raise e


if __name__ == "__main__":
    train()
