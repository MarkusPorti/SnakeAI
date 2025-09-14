import sys
from typing import Any

import pygame
import gymnasium as gym
from gymnasium.core import ActType

from env import SnakeEnvironment, Direction


class PygameVisualizer(gym.Wrapper):
    """
    A wrapper for the SnakeEnvironment that visualizes the game using Pygame.
    """
    metadata = {"render_modes": ["human"], "render_fps": 15}

    def __init__(self, env: SnakeEnvironment, block_size: int = 40, render_mode: str = "human"):
        super().__init__(env)
        self.env: SnakeEnvironment  # For type hinting

        self.block_size = block_size

        # Pygame is initialized on the first render call
        self.screen = None
        self.clock = None

        # Colors
        self.COLORS = {
            "background": (20, 30, 20),
            "grass": (40, 150, 40),
            "wall": (100, 100, 100),
            "food": (200, 50, 50),
            "snake_head": (50, 255, 50),
            "snake_body": (40, 200, 40),
        }

    def step(self, action: ActType):
        """
        Takes a step in the environment and renders the new state.
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.render()
        return obs, reward, terminated, truncated, info

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        """
        Resets the environment and renders the initial state.
        """
        obs, info = self.env.reset(seed=seed, options=options)
        self.render()
        return obs, info

    def render(self):
        """
        Renders the current state of the environment using Pygame.
        """
        if self.screen is None:
            pygame.init()
            # The +2 is for the walls on each side
            screen_width = (self.env.width + 2) * self.block_size
            screen_height = (self.env.height + 2) * self.block_size
            self.screen = pygame.display.set_mode((screen_width, screen_height))
            pygame.display.set_caption("Snake")
            self.clock = pygame.time.Clock()

        # Handle Pygame events (like closing the window)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                sys.exit()

        # --- Drawing ---
        self.screen.fill(self.COLORS["background"])

        # Draw walls and grass
        for x in range(self.env.width + 2):
            for y in range(self.env.height + 2):
                rect = pygame.Rect(
                    x * self.block_size,
                    y * self.block_size,
                    self.block_size,
                    self.block_size
                )
                if x == 0 or x == self.env.width + 1 or y == 0 or y == self.env.height + 1:
                    pygame.draw.rect(self.screen, self.COLORS["wall"], rect)
                else:
                    pygame.draw.rect(self.screen, self.COLORS["grass"], rect)


        # Draw the snake's body
        for part in self.env.snake.body:
            # +1 offset because of the walls
            rect = pygame.Rect(
                (part.x + 1) * self.block_size,
                (part.y + 1) * self.block_size,
                self.block_size,
                self.block_size,
            )
            pygame.draw.rect(self.screen, self.COLORS["snake_body"], rect)

        # Draw the snake's head
        head_rect = pygame.Rect(
            (self.env.snake.head.x + 1) * self.block_size,
            (self.env.snake.head.y + 1) * self.block_size,
            self.block_size,
            self.block_size,
        )
        pygame.draw.rect(self.screen, self.COLORS["snake_head"], head_rect)

        # Draw the food
        if self.env.food:
            food_rect = pygame.Rect(
                (self.env.food.x + 1) * self.block_size,
                (self.env.food.y + 1) * self.block_size,
                self.block_size,
                self.block_size,
            )
            pygame.draw.rect(self.screen, self.COLORS["food"], food_rect)

        # Update the display
        pygame.display.flip()

        # Control the frame rate
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        """
        Cleans up the Pygame window.
        """
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None


if __name__ == '__main__':
    # --- Example Usage ---

    # 1. Create the base environment (no visualization)
    # This is what you would use for training.
    base_env = SnakeEnvironment(width=10, height=10)

    # 2. Wrap the base environment with the PygameVisualizer
    # This is what you would use for validation or just to watch the agent play.
    visual_env = PygameVisualizer(base_env, block_size=30)

    # --- Run a simple loop with random actions to demonstrate ---
    print("Starting visualization with random actions...")
    episodes = 5
    for episode in range(episodes):
        obs, info = visual_env.reset()
        terminated = False
        score = 0
        while not terminated:
            action = Direction(visual_env.action_space.sample())

            obs, reward, terminated, truncated, info = visual_env.step(action)
            score += reward

            # The visual_env.step() call automatically handles rendering
            # and the clock tick, so no extra sleep is needed.

        print(f"Episode {episode + 1} finished. Final score: {score:.3f}")

    # 3. Important: Close the environment to shut down Pygame
    visual_env.close()
    print("Visualization closed.")