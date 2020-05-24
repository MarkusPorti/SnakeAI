import pygame
from random import randint
from Snake import Game

import numpy as np
from keras.optimizers import RMSprop
from keras.models import Sequential, Model
from keras import layers

import seaborn as sns
import matplotlib.pyplot as plt


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


class Agent:
    def __init__(self):
        # TODO load model
        self.model: Model = self.__create_model()
        self.game = Game(gui=True)

    def __create_model(self):
        _model = Sequential()
        _model.add(layers.Conv2D(20, (3, 3), activation='relu', input_shape=(20, 20, 1)))
        _model.add(layers.Conv2D(40, (3, 3), activation='relu'))
        _model.add(layers.Conv2D(30, (3, 3), activation='relu'))
        _model.add(layers.Flatten())
        _model.add(layers.Dense(256, activation='relu'))
        _model.add(layers.Dense(3))

        # model.add(Dense(units=400, activation='relu', input_dim=400))
        # model.add(Dropout(0.5))
        # model.add(Dense(units=128, activation='relu'))
        # model.add(Dropout(0.5))
        # model.add(Dense(units=128, activation='relu'))
        # model.add(Dropout(0.5))
        # model.add(Dense(units=16, activation='relu'))
        # model.add(Dense(units=3, activation='softmax'))

        # opt = Adam(0.0001)  # TODO: change it later to 0.0001/5
        _model.compile(RMSprop(), "MSE")
        return _model

    def __train_model(self, state, action, reward, next_state, running):
        # target = reward
        # if running:
        target = reward + 0.9 * np.amax(self.model.predict(next_state[3])[0])
        target_f = self.model.predict(state[3])
        target_f[0][action] = target
        self.model.fit(state[3], target_f, epochs=1, verbose=0)

    def train(self, epochs=10000):
        epsilon = 1.

        score_plot = []
        counter_plot = []
        max_score: int = 0

        # while game.snake.score < 10:
        for counter_games in range(epochs):
            print('Simulation ', counter_games, '\r', end='')
            self.game.reset_game()

            if epsilon > .1:
                # fine tune epsilon
                epsilon -= .9 / (5000 / 2)

            # run one game till snake dies
            while self.game.running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        quit()
                # game.clock.tick(144)

                # get old state
                state_old = self.game.get_state()

                # at the beginning more random, lately more advised actions
                if np.random.random() > epsilon:
                    # use prediction
                    prediction = self.model.predict(state_old[3])
                    final_move = np.argmax(prediction[0])
                else:
                    final_move = randint(0, 2)
                    # print(counter_games, ': Prediciton was ', prediction, ' -> ', final_move)

                state_new = self.game.step(final_move)
                reward = 0
                if not state_new[0]:
                    reward = -1
                elif state_new[2].score > state_old[2].score:
                    reward = len(self.game.snake.snake)

                # train short
                self.__train_model(state_old, final_move, reward, state_new, state_new[0])
                # end of one move
            # end of one game.. snake died

            # print current state of training
            if counter_games % 10 == 0:
                print('Game : ', counter_games)
                print('Highest score till now was:', max_score)
                print('---------------------------')

            if self.game.snake.score > max_score:
                max_score = self.game.snake.score

            score_plot.append(self.game.snake.score)
            counter_plot.append(counter_games)

        # save current weights of self.model
        self.model.save_weights('checkpoints/my_checkpoint')

        plot_seaborn(counter_plot, score_plot)


if __name__ == "__main__":
    agent = Agent()
    agent.train()
