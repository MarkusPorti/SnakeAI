from collections import namedtuple
from enum import Enum

import pygame
import numpy as np
from random import randint


pygame.init()
font = pygame.font.Font('arial.ttf', 25)


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


Point = namedtuple('Point', 'x, y')

# rgb colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)

BLOCK_SIZE = 20
SPEED = 180


class SnakeGameAI:
    def __init__(self, width=14, height=14):
        self.clock = pygame.time.Clock()
        self.width = width
        self.height = height

        self.reset()

        # init display
        pygame.display.set_caption("Snake")
        self.display = pygame.display.set_mode((self.width * BLOCK_SIZE, self.height * BLOCK_SIZE))
        self.__render()

    def reset(self):
        self.direction = Direction.RIGHT
        self.head = Point(self.width // 2, self.height // 2)
        self.snake = [self.head,
                      Point(self.head.x - 1, self.head.y),
                      Point(self.head.x - 2, self.head.y)]

        self.running = True
        self.score = 0
        self.food = None
        self.__generate_food()
        self.frame = 0

    def __generate_food(self):
        x = randint(0, self.width - 1)
        y = randint(0, self.height - 1)
        self.food = Point(x, y)
        if self.food in self.snake:
            self.__generate_food()

    def play_step(self, action):
        self.frame += 1

        # 1. pygame-step
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # 2. move
        self._move(action)
        self.snake.insert(0, self.head)

        # 3. check if game over
        reward = 0
        if self.is_collision() or self.frame > 100 * len(self.snake):
            self.running = False
            reward = -1
            return reward, not self.running, self.score

        # 4. place new food or just move
        if self.head == self.food:
            self.score += 1
            reward = self.score
            self.__generate_food()
        else:
            self.snake.pop()

        # 5. update ui and clock
        self.__render()
        self.clock.tick(SPEED)

        # 6. return reward, done and score
        return reward, not self.running, self.score

    def _move(self, action):
        # [straight, right, left]

        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]  # no change
        elif np.array_equal(action, [0, 1, 0]):
            new_dir = clock_wise[(idx + 1) % 4]  # right turn: r > d > l > u
        else:  # [0, 0, 1]
            new_dir = clock_wise[(idx - 1) % 4]  # left turn: r > u > l > d

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += 1
        elif self.direction == Direction.LEFT:
            x -= 1
        elif self.direction == Direction.DOWN:
            y += 1
        elif self.direction == Direction.UP:
            y -= 1

        self.head = Point(x, y)

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # hits boundary
        if pt.x >= self.width or pt.x < 0 or pt.y >= self.height or pt.y < 0:
            return True
        # hits itself
        if pt in self.snake[1:]:
            return True

        return False

    def get_state(self):
        board = np.ones((self.width + 2, self.height + 2))
        snake = np.asarray(self.snake)
        board[snake[1:, 0] + 1, snake[1:, 1] + 1] = 128  # body = 1
        if self.running:
            board[self.head[0] + 1][self.head[1] + 1] = 96  # head = 2/3
        board[:, 0] = 255  # left wall
        board[:, -1] = 255  # right wall
        board[0, :] = 255  # top wall
        board[-1, :] = 255  # bottom wall
        board[self.food[0] + 1][self.food[1] + 1] = 48  # food = 4/5
        return board.reshape(self.get_observation_shape())  # .reshape((1, (self.width + 2) * (self.height + 2)))

    def get_observation_shape(self):
        return 1, self.width + 2, self.height + 2

    def __render(self):
        self.display.fill(BLACK)

        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x * BLOCK_SIZE, pt.y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x * BLOCK_SIZE + 4, pt.y * BLOCK_SIZE + 4, 12, 12))

        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x * BLOCK_SIZE, self.food.y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))

        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()
