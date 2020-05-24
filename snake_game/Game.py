import pygame
from random import randint, random
from snake_game.Snake import Snake

import numpy as np
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.optimizers import RMSprop
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras import layers
import collections
import random

import seaborn as sns
import matplotlib.pyplot as plt

from threading import Thread


class Game:
    def __init__(self, width=20, height=20, gui=False):
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
        self.board = np.zeros(self.width * self.height, dtype=int)
        snake = np.array(self.snake.snake)
        snake = snake[:, 1] * 20 + snake[:, 0]
        self.board[snake[1:]] = 1
        self.board[0:20] = 1
        self.board[380:400] = 1
        if self.running:
            self.board[snake[0]] = 1
        self.board[self.food[1] * 20 + self.food[0]] = .5
        self.board = np.asarray(self.board).reshape(1, 20, 20, 1)
        self.board[:, 0] = 1
        self.board[:, -1] = 1
        return self.board


def create_model():
    model = Sequential()
    model.add(layers.Conv2D(20, (3, 3), activation='relu', input_shape=(20, 20, 1)))
    model.add(layers.Conv2D(40, (3, 3), activation='relu'))
    model.add(layers.Conv2D(30, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(3))

    # model.add(Dense(units=400, activation='relu', input_dim=400))
    # model.add(Dropout(0.5))
    # model.add(Dense(units=128, activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(units=128, activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(units=16, activation='relu'))
    # model.add(Dense(units=3, activation='softmax'))

    # opt = Adam(0.0001)  # TODO: change it later to 0.0001/5
    model.compile(RMSprop(), "MSE")
    return model


def remember(state, action, reward, next_state, running):
    memory.append((state, action, reward, next_state, running))


def train_short_memmory(state, action, reward, next_state, running):
    # target = reward
    # if running:
    target = reward + 0.9 * np.amax(model.predict(next_state[3])[0])
    target_f = model.predict(state[3])
    target_f[0][action] = target
    # prediciton = model.predict(next_state[3])
    # prediciton[0][action] = reward
    model.fit(state[3], target_f, epochs=1, verbose=0)


def train_long_memmory(batch_size):
    if len(memory) > batch_size:
        minibatch = random.sample(memory, batch_size)
    else:
        minibatch = memory
    for state, action, reward, next_state, running in minibatch:
        target = reward
        if running:
            target = reward + np.amax(model.predict(next_state[3])[0])
        target_f = model.predict(state[3])
        target_f[0][action] = target
        model.fit(state[3], target_f, epochs=1, verbose=0)
    memory.clear()


def plot_seaborn(array_counter, array_score):
    sns.set(color_codes=True)
    ax = sns.regplot(
        np.array([array_counter])[0],
        np.array([array_score])[0],
        color="b",
        x_jitter=.1,
        line_kws={'color': 'green'}
    )
    ax.set(xlabel='games', ylabel='score')
    plt.show()


def run():
    global model, memory
    model = create_model()
    memory = collections.deque(maxlen=3000)

    epsilon= 1.
    score_plot = []
    counter_plot = []
    counter_games: int = 0
    max_score: int = 0
    game = Game(gui=True)
    random_moves = True

    # for _ in range(2000):
    while game.snake.score < 10:
        print('Simulation ', counter_games, '\r', end='')
        game.reset_game()

        if epsilon > .1:
            # fine tune epsilon
            epsilon -= .9 / (5000 / 2)

        # run one game till snake dies
        while game.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
            # game.clock.tick(144)

            # get old state
            state_old = game.get_state()

            # at the beginning more random, lately more advised actions
            if np.random.random() > epsilon:
                # use prediction
                prediction = model.predict(state_old[3])
                final_move = np.argmax(prediction[0])
            else:
                final_move = randint(0, 2)
                # print(counter_games, ': Prediciton was ', prediction, ' -> ', final_move)

            state_new = game.step(final_move)
            reward = 0
            if not state_new[0]:
                reward = -1
            elif state_new[2].score > state_old[2].score:
                reward = len(game.snake.snake)

            # train short
            train_short_memmory(state_old, final_move, reward, state_new, state_new[0])
            # remember(state_old, final_move, reward, state_new, state_new[0])

        # train long
        # train_long_memmory(500)

        # print current state of training
        if counter_games % 10 == 0:
            print('Game : ', counter_games)
            print('Highest score till now was:', max_score)
            print('---------------------------')

        if game.snake.score > max_score:
            max_score = game.snake.score

        score_plot.append(game.snake.score)
        counter_plot.append(counter_games)
        counter_games += 1

    # save current weights of model
    model.save_weights('./checkpoints/my_checkpoint')

    plot_seaborn(counter_plot, score_plot)


if __name__ == "__main__":
    run()
