from dataclasses import dataclass
from enum import Enum
from random import randint
from typing import Any, SupportsFloat

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces
from gymnasium.core import ObsType, ActType, RenderFrame


class Direction(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

    def opposite(self) -> "Direction":
        if self.value < 2:
            return Direction(self.value + 2)
        return Direction(self.value - 2)


class FieldType(Enum):
    FOOD = 0
    SNAKE_HEAD = 1
    SNAKE_BODY = 2


@dataclass(frozen=True)
class FieldPoint:
    x: int
    y: int


class Snake:
    head: FieldPoint
    body: list[FieldPoint]
    direction: Direction

    def __init__(self, x: int, y: int):
        self.reset(x, y)

    def reset(self, x: int, y: int):
        self.head = FieldPoint(x=x, y=y)
        self.body = [
            FieldPoint(x=x - 1, y=y),
            FieldPoint(x=x - 2, y=y),
        ]
        self.direction = Direction.RIGHT

    def step(self, action: Direction, food: FieldPoint, width: int, height: int):
        # Move into the direction.
        self.direction = action

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

        # 3. Swallowed an apple? Good job.
        if self.head == food:
            return 1

        # Now check for any collisions
        self.body.pop()
        if (
            self.head.x == -1
            or self.head.y == -1
            or self.head.x == width
            or self.head.y == height
        ):
            # We are out of the board
            return -1

        if self.head in self.body:
            return -2

        # Nothing special happened
        return 0

    def __iter__(self):
        return iter([self.head] + self.body)


class SnakeEnvironment(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 2}

    # Fields for the actual Game / State
    snake: Snake
    food: FieldPoint | None
    score: int
    health: int

    # Fields for rendering
    window: pygame.Surface | None = None
    clock: pygame.time.Clock | None = None
    block_size: int = 30
    COLORS: dict[str, pygame.Color] = {
        "background": (20, 30, 20),
        "grass": (40, 150, 40),
        "wall": (100, 100, 100),
        "food": (200, 50, 50),
        "snake_head": (50, 255, 50),
        "snake_body": (40, 200, 40),
    }

    def __init__(self, width: int = 18, height: int = 18, render_mode: str = None):
        super().__init__()
        self.width = width
        self.height = height
        self.render_mode = render_mode
        self.window_width = self.width * self.block_size
        self.window_height = self.height * self.block_size

        self.action_space = spaces.Discrete(n=len(Direction))
        self.observation_space = spaces.MultiBinary(
            n=[len(FieldType), self.width, self.height]
        )

        self.snake = Snake(x=self.width // 2, y=self.height // 2)
        self.reset()

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        action = Direction(action)
        result = self.snake.step(action, self.food, self.width, self.height)

        self.health -= 1
        if result == 1:
            self.food = self._generate_food()
            self.health += 30
            self.score += 1

        reward = self.calc_reward(result)

        terminated = result < 0
        truncated = self.health <= 0
        return self._get_obs(head_in_wall=result==-1), reward, terminated, truncated, {}

    def calc_reward(self, step_result: int) -> float:
        reward = 0
        if step_result == 1:
            reward += self.score * 3
        elif step_result == 0:
            reward -= 0.0001
        elif step_result < 0:
            reward = -1
        return reward

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ObsType, dict[str, Any]]:
        super().reset(seed=seed)
        self.snake.reset(x=self.width // 2, y=self.height // 2)
        self.food = self._generate_food()

        self.score = 0
        self.health = 50

        return self._get_obs(), {}

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

    def _get_obs(self, head_in_wall: bool = False) -> ObsType:
        board = np.zeros((len(FieldType), self.width, self.height), dtype=np.int8)

        # Snake
        if not head_in_wall:
            board[FieldType.SNAKE_HEAD.value, self.snake.head.x, self.snake.head.y] = 1
        snake = np.asarray([(fp.x, fp.y) for fp in self.snake.body])
        board[FieldType.SNAKE_BODY.value, snake[:, 0], snake[:, 1]] = 1
        # Food
        board[FieldType.FOOD.value, self.food.x, self.food.y] = 1

        return board

    @staticmethod
    def action_mask(env: "SnakeEnvironment") -> np.ndarray:
        mask = np.ones((len(Direction),), dtype=np.bool)
        mask[env.snake.direction.opposite().value] = False
        return mask

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        if self.render_mode in ["human", "rgb_array"]:
            return self._render_frame()
        return None

    def _render_frame(self) -> RenderFrame | list[RenderFrame] | None:
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_width, self.window_height))
            pygame.display.set_caption("Snake")
            self.clock = pygame.time.Clock()

        # --- Drawing ---
        canvas = pygame.Surface((self.window_width, self.window_height))
        canvas.fill(self.COLORS["background"])

        # Draw the snake's body
        for part in self.snake.body:
            rect = pygame.Rect(
                part.x * self.block_size,
                part.y * self.block_size,
                self.block_size,
                self.block_size,
            )
            pygame.draw.rect(canvas, self.COLORS["snake_body"], rect)

        # Draw the snake's head
        head_rect = pygame.Rect(
            self.snake.head.x * self.block_size,
            self.snake.head.y * self.block_size,
            self.block_size,
            self.block_size,
        )
        pygame.draw.rect(canvas, self.COLORS["snake_head"], head_rect)

        # Draw the food
        food_rect = pygame.Rect(
            self.food.x * self.block_size,
            self.food.y * self.block_size,
            self.block_size,
            self.block_size,
        )
        pygame.draw.rect(canvas, self.COLORS["food"], food_rect)

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
            return None
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
