from dataclasses import dataclass
from enum import Enum
from typing import Any, SupportsFloat

import numpy as np
import gymnasium as gym
from random import randint

from gymnasium import spaces
from gymnasium.core import ObsType, ActType


class Direction(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


class FieldType(Enum):
    WALL = 0
    GRASS = 1
    FOOD = 2
    SNAKE_HEAD = 3
    SNAKE_BODY = 4


@dataclass(frozen=True)
class FieldPoint:
    x: int
    y: int


class Snake:
    head: FieldPoint
    body: list[FieldPoint]

    def __init__(self, x: int, y: int):
        self.reset(x, y)

    def reset(self, x: int, y: int):
        self.head = FieldPoint(x=x, y=y)
        self.body = [
            FieldPoint(x=x - 1, y=y),
            FieldPoint(x=x - 2, y=y),
        ]

    def step(self, action: Direction, food: FieldPoint, width: int, height: int):
        # Move into the direction.
        # 1. The old head becomes part of the body
        self.body.insert(0, self.head)

        # 2. Determine the new heads position
        if action == Direction.UP:
            self.head = FieldPoint(x=self.head.x, y=self.head.y - 1)
        elif action == Direction.RIGHT:
            self.head = FieldPoint(x=self.head.x + 1, y=self.head.y)
        elif action == Direction.DOWN:
            self.head = FieldPoint(x=self.head.x, y=self.head.y + 1)
        elif action == Direction.LEFT:
            self.head = FieldPoint(x=self.head.x - 1, y=self.head.y)
        else:
            raise ValueError(f"Invalid action. Must be one of {list(Direction)}")

        # 3. If no food was eaten: shorten the snakes tail
        if self.head != food:
            self.body.pop()
        else:
            # We swallowed an apple.
            # There is no chance this was a collision
            return 1

        # Now check for any collisions
        if (
            self.head.x == -1
            or self.head.y == -1
            or self.head.x == width
            or self.head.y == height
            or self.head in self.body
        ):
            # We are our of the board or have eaten our body
            return -1

        # Nothing special happened
        return 0

    def __iter__(self):
        return iter([self.head] + self.body)


class SnakeEnvironment(gym.Env):
    snake: Snake
    food: FieldPoint | None
    score: int

    def __init__(self, width: int = 18, height: int = 18):
        super().__init__()
        self.width = width
        self.height = height

        self.action_space = spaces.Discrete(n=len(Direction))
        self.observation_space = spaces.MultiBinary(
            n=[self.width + 2, self.height + 2, len(FieldType)]
        )

        self.snake = Snake(x=self.width // 2, y=self.height // 2)
        self.reset()

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:

        result = self.snake.step(action, self.food, self.width, self.height)
        reward = self.calc_reward(result)

        if result == 1:
            self.food = self._generate_food()

        terminated = result == -1
        return self.get_observation(), reward, terminated, False, {}

    @staticmethod
    def calc_reward(step_result: int) -> float:
        reward = 0.0
        if step_result == 1:
            reward += 1
        elif step_result == 0:
            reward += 0.001
        elif step_result == -1:
            reward = -20
        return reward

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ObsType, dict[str, Any]]:
        self.snake.reset(x=self.width // 2, y=self.height // 2)
        self.food = self._generate_food()

        self.score = 0

        return self.get_observation(), {}

    def _generate_food(self) -> FieldPoint:
        food = None
        while not food:
            food = FieldPoint(
                x=randint(0, self.width - 1),
                y=randint(0, self.height - 1),
            )
            if food in self.snake:
                food = None
        return food

    def get_observation(self) -> ObsType:
        board = np.zeros((self.width + 2, self.height + 2, len(FieldType)))

        # Board
        board[0, :, FieldType.WALL.value] = 1
        board[:, -1, FieldType.WALL.value] = 1
        board[-1, :, FieldType.WALL.value] = 1
        board[:, 0, FieldType.WALL.value] = 1
        board[1:-1, 1:-1, FieldType.GRASS.value] = 1
        # Snake
        board[self.snake.head.x + 1, self.snake.head.y + 1, FieldType.SNAKE_HEAD.value] = 1
        snake = np.asarray([(fp.x, fp.y) for fp in self.snake.body])
        board[snake[:, 0] + 1, snake[:, 1] + 1, FieldType.SNAKE_BODY.value] = 1
        # Food
        board[self.food.x, self.food.y, FieldType.FOOD.value] = 1

        return board
