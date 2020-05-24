import pygame
import numpy as np
from random import randint


class Game:
    def __init__(self, width=18, height=18, gui=False):
        self.pixel = 20
        self.food_color = (255, 0, 0)
        self.clock = pygame.time.Clock()
        self.width = width
        self.height = height
        self.board = np.ones((width, height), dtype=int)  # complete Board
        self.moves_left = width * height
        self.snake = Snake(self.width, self.height, self.pixel)
        self.food = ()
        self.generate_food()
        self.running = True
        self.gui = gui
        if self.gui:
            self.render_init()

    def reset_game(self):
        self.moves_left = self.width * self.height
        self.snake.reset()
        self.food = []
        self.generate_food()
        self.running = True

    def render_init(self):
        pygame.init()
        pygame.display.set_caption("Snake")
        self.font = pygame.font.SysFont("Comic Sans Ms", 16)
        self.dis = pygame.display.set_mode((self.width * self.pixel, self.height * self.pixel))
        self.render()

    def render(self):
        self.dis.fill((255, 255, 255))
        pygame.draw.rect(self.dis, self.food_color,
                         [self.food[0] * self.pixel, self.food[1] * self.pixel, self.pixel, self.pixel])
        self.snake.render(self.dis)

        text_score = self.font.render("Score: " + str(self.snake.score), True, (0, 0, 0))
        self.dis.blit(text_score, (5, 0))
        text_score = self.font.render("Moves left: " + str(self.moves_left), True, (0, 0, 0))
        self.dis.blit(text_score, (5, 20))
        pygame.display.update()

    def step(self, move):
        result = self.snake.step(move, self.food)
        if result == 0:
            self.moves_left -= 1

        if result == -1 or self.moves_left <= 0:
            self.running = False
        elif result == 1:
            self.moves_left = self.width * self.height
            self.generate_food()
        if self.gui:
            # Thread(target=self.render).start()
            self.render()
        return self.get_state()

    def generate_food(self):
        food = ()
        while not food:
            food = (randint(0, self.width - 1), randint(0, self.height - 1))
            if food in self.snake:
                food = ()
        self.food = food

    def get_state(self):
        return self.running, self.moves_left, self.snake, self.__get_map()

    def __get_map(self):
        self.board = np.zeros((self.width + 2, self.height + 2))
        snake = np.asarray(self.snake.snake)
        self.board[snake[1:, 1], snake[1:, 0]] = 1  # body = 1
        if self.running:
            self.board[self.snake.snake[0]] = 1  # head = 1
        self.board[:, 0] = 1  # left wall
        self.board[:, -1] = 1  # right wall
        self.board[0, :] = 1  # top wall
        self.board[-1, :] = 1  # bottom wall
        self.board[self.food[1]][self.food[0]] = .5  # food = 0.5
        return self.board.reshape((1, self.width + 2, self.height + 2, 1))


class Snake:
    def __init__(self, width, height, pixel=20, x=10, y=10):
        self.pixel = pixel
        self.width = width
        self.height = height
        self.color_head = (0, 150, 0)
        self.color_body = (25, 200, 25)

        self.score = 0

        self.direction = 0
        self.snake = [(x, y)]
        for i in range(1, 3):
            self.snake.append((x, y + i))

    def reset(self, x=10, y=10):
        self.score = 0
        self.direction = 0
        self.snake = [(x, y)]
        for i in range(1, 3):
            self.snake.append((x, y + i))

    def step(self, move, food):
        # 0 - straight
        # 1 - left
        # 2 - right
        if move == 1:
            if self.direction > 0:
                self.direction -= 1
            else:
                self.direction = 3

        if move == 2:
            if self.direction < 3:
                self.direction += 1
            else:
                self.direction = 0

        head = self.snake[0]
        if self.direction == 0:
            head = (self.snake[0][0], self.snake[0][1] - 1)
        elif self.direction == 1:
            head = (self.snake[0][0] + 1, self.snake[0][1])
        elif self.direction == 2:
            head = (self.snake[0][0], self.snake[0][1] + 1)
        elif self.direction == 3:
            head = (self.snake[0][0] - 1, self.snake[0][1])
        self.snake.insert(0, head)

        if self.eat(food):
            self.score += 1
            return 1
        else:
            self.snake.pop()

        if self.check_collision():
            return -1
        else:
            return 0

    def eat(self, food):
        return self.snake[0] == food

    def check_collision(self):
        return (self.snake[0][0] == -1 or
                self.snake[0][1] == -1 or
                self.snake[0][0] == self.width or
                self.snake[0][1] == self.height or
                self.snake[0] in self.snake[1:])

    def render(self, display):
        for i, point in enumerate(self.snake):
            if i == 0:
                pygame.draw.rect(display, self.color_head,
                                 [self.pixel * point[0], self.pixel * point[1], self.pixel, self.pixel])
            else:
                pygame.draw.rect(display, self.color_body,
                                 [self.pixel * point[0], self.pixel * point[1], self.pixel, self.pixel])

    def __iter__(self):
        return iter(self.snake)
