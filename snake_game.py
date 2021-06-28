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
    def __init__(self, width=20, height=20):
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
            reward = 1
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
        dangers = self.__get_dangers(self.head, self.direction)

        if self.direction == Direction.RIGHT:
            # straight, right, left
            point_straight = Point(self.head.x + 2, self.head.y)
            point_right = Point(self.head.x + 1, self.head.y + 1)
            point_left = Point(self.head.x + 1, self.head.y - 1)
            dir_straight = Direction.RIGHT
            dir_right = Direction.DOWN
            dir_left = Direction.UP
        if self.direction == Direction.LEFT:
            # straight, right, left
            point_straight = Point(self.head.x - 2, self.head.y)
            point_right = Point(self.head.x - 1, self.head.y - 1)
            point_left = Point(self.head.x - 1, self.head.y + 1)
            dir_straight = Direction.LEFT
            dir_right = Direction.UP
            dir_left = Direction.DOWN
        if self.direction == Direction.UP:
            # straight, right, left
            point_straight = Point(self.head.x, self.head.y - 2)
            point_right = Point(self.head.x + 1, self.head.y - 1)
            point_left = Point(self.head.x - 1, self.head.y - 1)
            dir_straight = Direction.UP
            dir_right = Direction.RIGHT
            dir_left = Direction.LEFT
        if self.direction == Direction.DOWN:
            # straight, right, left
            point_straight = Point(self.head.x, self.head.y + 2)
            point_right = Point(self.head.x - 1, self.head.y + 1)
            point_left = Point(self.head.x + 1, self.head.y + 1)
            dir_straight = Direction.DOWN
            dir_right = Direction.LEFT
            dir_left = Direction.RIGHT

        dangers_straight = self.__get_dangers(point_straight, dir_straight)
        dangers_right = self.__get_dangers(point_right, dir_right)
        dangers_left = self.__get_dangers(point_left, dir_left)

        state = [
            # Danger 1.
            dangers[0], dangers[1], dangers[2],

            # Move direction
            self.direction == Direction.RIGHT,
            self.direction == Direction.LEFT,
            self.direction == Direction.UP,
            self.direction == Direction.DOWN,

            # Danger 2.
            dangers_straight[0], dangers_straight[1], dangers_straight[2],
            dangers_right[0], dangers_right[1], dangers_right[2],
            dangers_left[0], dangers_left[1], dangers_left[2],

            # Food location
            self.food.x < self.head.x,  # food left
            self.food.x > self.head.x,  # food right
            self.food.y < self.head.y,  # food up
            self.food.y > self.head.y  # food down
        ]

        return np.array(state, dtype=int)

    def __get_dangers(self, point, direction):
        point_r = Point(point.x + 1, point.y)
        point_l = Point(point.x - 1, point.y)
        point_u = Point(point.x, point.y - 1)
        point_d = Point(point.x, point.y + 1)

        # Danger straight, right, left
        if direction == Direction.RIGHT:
            return self.is_collision(point_r), self.is_collision(point_d), self.is_collision(point_u)
        if direction == Direction.LEFT:
            return self.is_collision(point_l), self.is_collision(point_u), self.is_collision(point_d)
        if direction == Direction.UP:
            return self.is_collision(point_u), self.is_collision(point_r), self.is_collision(point_l)
        if direction == Direction.DOWN:
            return self.is_collision(point_d), self.is_collision(point_l), self.is_collision(point_r)

    def __render(self):
        self.display.fill(BLACK)

        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1,
                             pygame.Rect(pt.x * BLOCK_SIZE, pt.y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x * BLOCK_SIZE + 4, pt.y * BLOCK_SIZE + 4, 12, 12))

        pygame.draw.rect(self.display, RED,
                         pygame.Rect(self.food.x * BLOCK_SIZE, self.food.y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))

        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()
