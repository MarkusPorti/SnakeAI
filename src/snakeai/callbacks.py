import numpy as np
import torch as th
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import HParam
from stable_baselines3.common.logger import Video

from snakeai.env import SnakeEnvironment


class VideoRecorderCallback(BaseCallback):
    def __init__(
        self,
        eval_env: gym.Env,
        render_freq: int,
        n_eval_episodes: int = 1,
        deterministic: bool = True,
        fps: int = 2,
    ):
        """
        Records a video of an agent's trajectory traversing ``eval_env`` and logs it to TensorBoard

        :param eval_env: A gym environment from which the trajectory is recorded
        :param render_freq: Render the agent's trajectory every eval_freq call of the callback.
        :param n_eval_episodes: Number of episodes to render
        :param deterministic: Whether to use deterministic or stochastic policy
        """
        super().__init__()
        assert eval_env.render_mode == "rgb_array", (
            f"The Environment has to be in RGB render mode. Was in {eval_env.render_mode}"
        )
        self._eval_env = eval_env
        self._render_freq = render_freq
        self._n_eval_episodes = n_eval_episodes
        self._deterministic = deterministic
        self._fps = fps

    def _on_step(self) -> bool:
        if self.n_calls % self._render_freq == 0:
            screens = []

            obs, _ = self._eval_env.reset()
            screens.append(self._eval_env.render().transpose(2, 0, 1))
            done = False
            while not done:
                action, _ = self.model.predict(
                    obs,
                    deterministic=self._deterministic,
                    action_masks=SnakeEnvironment.action_mask(self._eval_env.unwrapped),
                )
                obs, reward, terminated, truncated, info = self._eval_env.step(action)
                screens.append(self._eval_env.render().transpose(2, 0, 1))
                done = terminated or truncated

            self.logger.record(
                "trajectory/video",
                Video(th.from_numpy(np.asarray([screens])), fps=self._fps),
                exclude=("stdout", "log", "json", "csv"),
            )
        return True


class HParamCallback(BaseCallback):
    """
    Saves the hyperparameters and metrics at the start of the training, and logs them to TensorBoard.
    """

    def _on_training_start(self) -> None:
        self.model: PPO
        hparam_dict = {
            "algorithm": self.model.__class__.__name__,
            "learning_rate": self.model.learning_rate,
            "n_steps": self.model.n_steps,
            "batch_size": self.model.batch_size,
            "n_epochs": self.model.n_epochs,
            # "clip_range": self.model.clip_range,
            "ent_coef": self.model.ent_coef,
            "gamma": self.model.gamma,
        }
        # define the metrics that will appear in the `HPARAMS` Tensorboard tab by referencing their tag
        # Tensorbaord will find & display metrics from the `SCALARS` tab
        metric_dict = {
            "rollout/ep_len_mean": 0,
            "train/clip_range": 0.0,
            "train/loss": 0.0,
            "train/entropy_loss": 0.0,
            "train/value_loss": 0.0,
        }
        self.logger.record(
            "hparams",
            HParam(hparam_dict, metric_dict),
            exclude=("stdout", "log", "json", "csv"),
        )

    def _on_step(self) -> bool:
        return True
