import os.path
import typing as ty

import click
import gymnasium as gym
import mlflow
import torch
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import ProgressBarCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import Logger
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

from snakeai.callbacks import VideoRecorderCallback, HParamCallback
from snakeai.env import SnakeEnvironment
from snakeai.logger import create_logger
from snakeai.model import CNNBoardFeatureExtractor

mlflow.set_tracking_uri("http://localhost:5000")


@click.group()
def cli():
    pass


def get_params() -> dict[str, ty.Any]:
    return dict(
        policy_kwargs=dict(
            features_extractor_class=CNNBoardFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=256),
            activation_fn=torch.nn.ReLU,
            net_arch=dict(pi=[32, 16], vf=[32, 16]),
        ),
        n_steps=128,
        learning_rate=1e-4,
        batch_size=128,
        n_epochs=10,
        clip_range=0.15,
        ent_coef=0.01,
        gamma=0.995,
    )


def create_env(render_mode: str | None = None) -> gym.Env:
    env = SnakeEnvironment(width=10, height=10, render_mode=render_mode)
    env = ActionMasker(env, "action_mask")
    env = Monitor(env)
    return env


def create_model(env: gym.Env, params: dict[str, ty.Any], logger: Logger) -> MaskablePPO:
    model = MaskablePPO(policy=MaskableActorCriticPolicy, env=env, **params)
    model.set_logger(logger)
    mlflow.log_params(params)
    return model


@cli.command()
@click.option("--n_envs", default=1, help="Number of parallel environments to use")
@click.argument("experiment")
def train(n_envs: int, experiment: str):
    sb3_logger = create_logger(experiment, "./runs", True)

    exp = mlflow.get_experiment_by_name(experiment)
    exp_id = exp.experiment_id if exp else mlflow.create_experiment(experiment)

    with mlflow.start_run(experiment_id=exp_id) as run:
        env = make_vec_env(create_env, n_envs=n_envs, vec_env_cls=SubprocVecEnv if n_envs > 1 else DummyVecEnv)
        params = get_params()
        model = create_model(env, params, sb3_logger)

        print(model.policy)

        eval_env = create_env(render_mode="rgb_array")
        hparams = HParamCallback()
        video_recorder = VideoRecorderCallback(eval_env, render_freq=2048)
        progress_bar = ProgressBarCallback()

        trained_model = model.learn(
            # total_timesteps=1024,
            total_timesteps=1_024_000,
            callback=[hparams, video_recorder, progress_bar],
            log_interval=4,
        )

        trained_model.save(os.path.join(sb3_logger.get_dir(), "model"))


@cli.command()
@click.argument("experiment")
def validate(experiment: str):
    env = create_env(render_mode="human")

    model = MaskablePPO.load(os.path.join("runs", experiment, "model"), env=env)

    obs, _ = env.reset()
    env.render()
    done = False
    while not done:
        action, _ = model.predict(
            obs,
            deterministic=True,
            action_masks=SnakeEnvironment.action_mask(env.unwrapped),
        )
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        done = terminated or truncated


if __name__ == "__main__":
    cli()
