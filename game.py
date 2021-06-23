import pygame
import numpy as np
from random import randint
from snake import Snake


class Game:
    def __init__(self, width=18, height=18, gui=False):
        self.pixel = 20
        self.food_color = (255, 0, 0)
        self.clock = pygame.time.Clock()
        self.width = width
        self.height = height
        self.board = np.ones((width, height), dtype=int)  # complete Board
        # self.moves_left = width * height
        self.snake = Snake(self.width, self.height, self.pixel)
        self.food = ()
        self.generate_food()
        self.running = True
        self.gui = gui
        if self.gui:
            self.render_init()

    def reset_game(self):
        # self.moves_left = self.width * self.height
        self.snake.reset()
        self.food = []
        self.generate_food()
        self.running = True
        return self.get_state(0)

    def render_init(self):
        pygame.init()
        pygame.display.set_caption("Snake")
        self.dis = pygame.display.set_mode((self.width * self.pixel, self.height * self.pixel))
        self.font = pygame.font.SysFont("Comic Sans Ms", 16)
        self.render()

    def render(self):
        # self.clock.tick(120)
        self.dis.fill((255, 255, 255))
        pygame.draw.rect(self.dis, self.food_color,
                         [self.food[0] * self.pixel, self.food[1] * self.pixel, self.pixel, self.pixel])
        self.snake.render(self.dis)

        # text_score = self.font.render("Score: " + str(self.snake.score), True, (0, 0, 0))
        # self.dis.blit(text_score, (5, 0))
        # text_score = self.font.render("Moves left: " + str(self.moves_left), True, (0, 0, 0))
        # self.dis.blit(text_score, (5, 20))
        pygame.display.update()

    def step(self, move):
        result = self.snake.step(move, self.food)

        reward = 0.001
        if result == -1:
            reward = -10
            self.running = False
        elif result == 1:
            reward = 1
            self.generate_food()

        if self.gui:
            # Thread(target=self.render).start()
            self.render()
        return self.get_state(reward)

    def step_random(self):
        return np.random.randint(len(self.snake.possible_actions))

    def generate_food(self):
        food = ()
        while not food:
            food = (randint(0, self.width - 1), randint(0, self.height - 1))
            if food in self.snake:
                food = ()
        self.food = food

    def get_state(self, reward):
        return self.__get_map(), reward, not self.running

    def __get_map(self):
        self.board = np.zeros((self.width + 2, self.height + 2))
        snake = np.asarray(self.snake.snake)
        self.board[snake[1:, 0] + 1, snake[1:, 1] + 1] = 1  # body = 1
        if self.running:
            snake = self.snake.snake[0]
            self.board[snake[0] + 1][snake[1] + 1] = 2 / 3  # head = 2/3
        self.board[:, 0] = 1  # left wall
        self.board[:, -1] = 1  # right wall
        self.board[0, :] = 1  # top wall
        self.board[-1, :] = 1  # bottom wall
        self.board[self.food[0] + 1][self.food[1] + 1] = 4 / 5  # food = 4/5
        return self.board.reshape(
            (1, self.width + 2, self.height + 2, 1))  # .reshape((1, (self.width + 2) * (self.height + 2)))

    def get_input_shape(self):
        return self.width + 2, self.height + 2, 1
