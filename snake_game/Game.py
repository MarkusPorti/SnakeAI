import pygame
from random import randint, random
from snake_game.Snake import Snake

import numpy as np
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
import collections
import random

import seaborn as sns
import matplotlib.pyplot as plt

from threading import Thread


class Game:
    def __init__(self, width=20, height=20, gui=False):
        self.gui = gui
        self.pixel = 20
        self.food = []
        self.running = True
        self.board = {'width': width, 'height': height}
        self.moves_left = width * height
        self.food_color = (255, 0, 0)
        self.clock = pygame.time.Clock()
        self.snake = self.get_new_snake()
        self.generate_food()
        if self.gui:
            self.render_init()

    def get_new_snake(self):
        return Snake(self.board['width'], self.board['height'], self.pixel)

    def calc_moves_left(self):
        return self.board['width'] * self.board['height']

    def reset_game(self):
        self.food = []
        self.generate_food()
        self.running = True
        self.moves_left = self.calc_moves_left()
        self.snake = self.get_new_snake()

    def render_init(self):
        pygame.init()
        pygame.display.set_caption("Snake")
        self.font = pygame.font.SysFont("Comic Sans Ms", 16)
        self.dis = pygame.display.set_mode((self.board["width"] * self.pixel, self.board["height"] * self.pixel))
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

    def step(self):
        result = self.snake.step(self.food)
        if result == 0:
            self.moves_left -= 1

        if result == -1 or self.moves_left <= 0:
            self.running = False
        elif result == 1:
            self.moves_left = self.board["width"] * self.board["height"]
            self.generate_food()
        if self.gui:
            Thread(target=self.render).start()
            # self.render()
        return self.get_state()

    def generate_food(self):
        food = []
        while food == []:
            food = [randint(0, self.board["width"] - 1), randint(0, self.board["height"] - 1)]
            if food in self.snake:
                food = []
        self.food = food

    def get_state(self):
        return self.running, self.moves_left, self.snake, self.__get_map()

    def __get_map(self):
        map = []
        for w in range(self.board['width']):
            map.append([])
            for h in range(self.board['height']):
                map[w].append(0)
                if [w, h] in self.snake:
                    map[w][h] = 1
                elif [w, h] == self.food:
                    map[w][h] = 2
                else:
                    map[w][h] = 0
        return np.array(map).flatten()


def create_model():
    model = Sequential()
    model.add(Dense(units=400, activation='relu', input_dim=400))
    model.add(Dense(units=200, activation='relu'))
    model.add(Dense(units=200, activation='relu'))
    model.add(Dense(units=4, activation='softmax'))
    opt = Adam(0.01)  # TODO: change it later to 0.0001/5
    model.compile(loss='mse', optimizer=opt)
    return model


def remember(state, action, reward, next_state, running):
    memory.append((state, action, reward, next_state, running))


def train_short_memmory(state, action, reward, next_state, running):
    target = reward
    if running:
        target = reward + np.amax(model.predict(next_state[3].reshape((1, 400))))
    target_f = model.predict(state[3].reshape((1, 400)))
    target_f[0][np.argmax(action)] = target
    model.fit(state[3].reshape((1, 400)), target_f, epochs=1, verbose=0)


def train_long_memmory(batch_size):
    if len(memory) > batch_size:
        minibatch = random.sample(memory, batch_size)
    else:
        minibatch = memory
    for state, action, reward, next_state, running in minibatch:
        target = reward
        if running:
            target = reward + np.amax(model.predict(np.array([next_state[3]]))[0])
        target_f = model.predict(np.array([state[3]]))
        target_f[0][np.argmax(action)] = target
        model.fit(np.array([state[3]]), target_f, epochs=1, verbose=0)


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


def run(episodes=300):
    global model, memory
    model = create_model()
    memory = collections.deque(maxlen=3000)

    score_plot = []
    counter_plot = []
    counter_games: int = 0
    max_score: int = 0
    game = Game(gui=False)

    # for counter_games in range(episodes):
    while game.snake.score < 100:

        print('Simulation ', counter_games, ' out of ', str(episodes), '\r', end='')
        game.reset_game()

        # run one game till snake dies
        while game.running:
            # game.clock.tick(60)

            # get old state
            state_old = game.get_state()

            # at the beginning more random, lately more advised actions
            if randint(0, 1) < 1 - (counter_games * 1 / 25):
                final_move = randint(0, 3)
            else:
                prediction = model.predict(state_old[3].reshape((1, 400)))
                final_move = np.argmax(prediction[0])

            game.snake.move(final_move)
            state_new = game.step()
            reward = 0.1
            if not state_new[0]:
                reward = -1
            elif state_new[2].score > state_old[2].score:
                reward = 1

            # train short
            train_short_memmory(state_old, final_move, reward, state_new, state_new[0])
            remember(state_old, final_move, reward, state_new, state_new[0])

        # train long
        train_long_memmory(500)

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
