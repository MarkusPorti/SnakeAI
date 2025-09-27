import os
import typing as ty

import mlflow
import numpy as np
from stable_baselines3.common.logger import (
    KVWriter,
    Logger,
    HumanOutputFormat,
    TensorBoardOutputFormat,
)
from stable_baselines3.common.utils import get_latest_run_id


class MLflowOutputFormat(KVWriter):
    """
    Dumps key/value pairs into MLflow's numeric format.
    """

    def write(
        self,
        key_values: dict[str, ty.Any],
        key_excluded: dict[str, ty.Union[str, tuple[str, ...]]],
        step: int = 0,
    ) -> None:
        for (key, value), (_, excluded) in zip(
            sorted(key_values.items()), sorted(key_excluded.items())
        ):
            if excluded is not None and "mlflow" in excluded:
                continue

            if isinstance(value, np.ScalarType):
                if not isinstance(value, str):
                    mlflow.log_metric(key, value, step, synchronous=False)


def create_logger(
    tb_experiment: str, tb_base_path: str = "./runs", reset_num_timesteps: bool = True
) -> Logger:
    latest_run_id = get_latest_run_id(tb_base_path, tb_experiment)
    if not reset_num_timesteps:
        # Continue training in the same directory
        latest_run_id -= 1
    save_path = os.path.join(tb_base_path, f"{tb_experiment}_{latest_run_id + 1}")
    os.makedirs(save_path, exist_ok=True)

    return Logger(
        save_path,
        [
            # HumanOutputFormat(sys.stdout),
            HumanOutputFormat(os.path.join(save_path, "log.txt")),
            TensorBoardOutputFormat(save_path),
            MLflowOutputFormat(),
        ],
    )
