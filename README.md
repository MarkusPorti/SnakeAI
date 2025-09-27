# SnakeAI
> Playing Snake using Reinforcement Learning.

The goal of this project is to train a Reinforcement Learning Agent capable of playing the classic Snake game.
The observation space is an image-like representation of the screen, encoding the snakes head, body and food.
The action space consists of four possible actions: `UP`, `RIGHT`, `DOWN` and `LEFT`.

## Roadmap

- [x] Game Simulation as [gymnasium](https://gymnasium.farama.org/) Environment
- [x] Training using [SB3](https://stable-baselines3.readthedocs.io/en/master/index.html) Algorithms
- [x] Use Action-Masks for preventing illegal moves (turning the opposite direction)
- [x] Experiment-Tracking using [Tensorboard](https://www.tensorflow.org/tensorboard) and [MLflow](https://mlflow.org/docs/latest/ml/)
- [ ] Log Evaluation Videos to MLflow.
- [ ] TODO: Get the Agent to learn a decent strategy

Some ideas:
- Improve the reward function
- Use multiple apples at the beginning, to increase the chance of eating one and gaining reward.
- Use CNNs instead of a simple flattening of the observation space
- Find a good architecture for the policy

## Project Setup

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/)
2. Run
```bash
uv sync --dev
```

### Experiment Tracking

The experiment is tracked using MLflow and Tensorboard.  
MLflow has a better overview over Hyperparameters, but currently doesn't support the evaluation Videos.
To show these, you can use Tensorboard.

You need to start a local MLflow Tracking Server, before running an experiment.  
You can simply use the Docker Configuration.  
Make sure to create a `.env` file in [mlflow](./mlflow) before starting the containers.

```bash
cd mlflow | docker compose up -d
```

To start the Tensorboard:
```bash
tensorboard --logdir runs
```

### Training

To start training and run an experiment, use the following CLI:

```bash
uv run snakeai train --n_envs=1 MY_EXPERIMENT
```

### Validation

To load a trained model and let it play, run

```bash
uv run snakeai validate MY_EXPERIMENT_1
```