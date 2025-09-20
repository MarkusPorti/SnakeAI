import os.path
import pathlib

import click
import gymnasium as gym
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import ProgressBarCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure, Logger
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import get_latest_run_id
from stable_baselines3.common.vec_env import SubprocVecEnv

from snakeai.callbacks import VideoRecorderCallback, HParamCallback
from snakeai.env import SnakeEnvironment


@click.group()
def cli():
    pass


def create_env(render_mode: str | None = None) -> gym.Env:
    env = SnakeEnvironment(width=10, height=10, render_mode=render_mode)
    env = ActionMasker(env, "action_mask")
    env = Monitor(env)
    return env


def create_logger(
    experiment: str, base_path: str = "./runs", reset_num_timesteps: bool = True
) -> Logger:
    latest_run_id = get_latest_run_id(base_path, experiment)
    if not reset_num_timesteps:
        # Continue training in the same directory
        latest_run_id -= 1
    save_path = os.path.join(base_path, f"{experiment}_{latest_run_id + 1}")
    return configure(save_path, ["log", "tensorboard"])


@cli.command()
@click.option("--n_envs", default=1, help="Number of parallel environments to use")
@click.argument("experiment")
def train(n_envs: int, experiment: str):
    env = make_vec_env(create_env, n_envs=n_envs, vec_env_cls=SubprocVecEnv)

    policy_kwargs = dict(
        # features_extractor_class=CNNBoardFeatureExtractor,
        # features_extractor_kwargs=dict(features_dim=32),
        # activation_fn=torch.nn.ReLU,
        net_arch=dict(pi=[32, 16], vf=[32, 16]),
    )
    model = MaskablePPO(
        policy=MaskableActorCriticPolicy,
        policy_kwargs=policy_kwargs,
        env=env,
        n_steps=128,
        learning_rate=1e-4,
        batch_size=64,
        n_epochs=10,
        clip_range=0.15,
        ent_coef=0.01,
        gamma=0.999,
    )
    model.set_logger(create_logger(experiment))

    print(model.policy)

    eval_env = create_env(render_mode="rgb_array")
    hparams = HParamCallback()
    video_recorder = VideoRecorderCallback(eval_env, render_freq=5000)
    progress_bar = ProgressBarCallback()

    trained_model = model.learn(
        # total_timesteps=1024,
        total_timesteps=128_000,
        callback=[hparams, video_recorder, progress_bar],
        log_interval=8,
    )

    model_name = pathlib.Path(trained_model.logger.get_dir()).name
    trained_model.save(f"./models/{model_name}")


@cli.command()
@click.argument("model")
def validate(model: str):
    env = create_env(render_mode="human")

    model = MaskablePPO.load(f"./models/{model}", env=env)

    obs, _ = env.reset()
    env.render()
    done = False
    while not done:
        action, _ = model.predict(
            obs, deterministic=True, action_masks=SnakeEnvironment.action_mask(env.unwrapped)
        )
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        done = terminated or truncated


if __name__ == '__main__':
    cli()
