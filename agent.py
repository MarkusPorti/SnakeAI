import pygame
from random import randint
from game import Game

import numpy as np
import tensorflow as tf
from keras.optimizers import Adam
from keras.optimizers import RMSprop
from keras.optimizers import SGD
from keras.models import Sequential, Model
from keras.layers.core import Dense, Flatten
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras import layers

import matplotlib.pyplot as plt


class Agent:
    def __init__(self):
        # TODO load model
        self.game = Game(12, 8, gui=True)
        self.model: Model = self.__create_model()
        self.losses = []

    def __create_model(self):
        _model = Sequential()
        _model.add(layers.Convolution2D(16, (3, 3), activation='relu',
                                        input_shape=(self.game.width + 2, self.game.height + 2, 1)))
        _model.add(layers.Convolution2D(32, (1, 1), activation='relu'))
        _model.add(Flatten())
        _model.add(Dense(256, activation='relu'))
        _model.add(Dense(3))
        _model.compile(RMSprop(), 'MSE')
        return _model

    def train(self, epochs=10000):
        epsilon = 1.

        score_plot = []
        counter_plot = []
        max_score: int = 0

        for counter_games in range(epochs):
            print('Simulation: ', counter_games, '\tScore: ', max_score, end='\r')
            self.game.reset_game()

            if epsilon > .1:
                # fine tune epsilon
                epsilon -= .9 / (epochs / 2)
            # else:
            #     epsilon = 0

            # run one game till snake dies
            while self.game.running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        tmp = range(0, len(self.losses))
                        plt.plot(tmp, self.losses)
                        plt.show()
                        pygame.quit()
                        quit()

                # get old state
                state_old = self.game.get_state()

                # at the beginning more random, lately more advised actions
                if np.random.random() > epsilon:
                    # use prediction
                    prediction = self.model.predict(state_old[2])
                    final_move = np.argmax(prediction[0])
                else:
                    final_move = randint(0, 2)

                state_new = self.game.step(final_move)
                reward = 0
                if not state_new[0]:
                    reward = -1
                elif state_new[1].score > state_old[1].score:
                    reward = 2

                # if reward != 0:
                self.__train_model(state_old, final_move, reward, state_new)

            if self.game.snake.score > max_score:
                max_score = self.game.snake.score

            # score_plot.append(self.game.snake.score)
            # counter_plot.append(counter_games)

        # save current weights of self.model
        self.model.save_weights('checkpoints/my_checkpoint')

        # plt.scatter(counter_plot, score_plot)
        # plt.draw()
        # # plot results
        # plt.show()

        # # plot losses
        # tmp = range(0, len(self.losses))
        # plt.plot(tmp, self.losses)
        # plt.show()

    def __train_model(self, game_state_old, action, reward, game_state_new):
        gamma = 0.9
        loss = False
        epochs = 3
        train_x = game_state_old[2]  # Eingabe
        train_y = self.model.predict(game_state_old[2])  # erwarteter Wert bei jeweiliger Eingabe
        if reward < 0:
            train_y[:, action] = reward
        else:
            optimal_future_val = self.model.predict(game_state_new[2])[0].max()
            train_y[0][action] = reward + gamma * optimal_future_val
        self.model.train_on_batch(train_x, train_y)

        # train_y = np.array(tf.nn.softmax(train_y))
        # if loss:
        #     self.losses = self.losses + history.history['loss']

    def play_manual(self, rounds=1):
        for counter_games in range(rounds):
            print('Game ', counter_games, '\r', end='')
            self.game.reset_game()

            # run one game till snake dies
            while self.game.running:
                self.game.clock.tick(5)

                final_move = 0
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        quit()

                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_UP:
                            final_move = 0
                        elif event.key == pygame.K_LEFT:
                            final_move = 1
                        elif event.key == pygame.K_RIGHT:
                            final_move = 2
                        break

                self.game.step(final_move)

        pygame.quit()
        quit()
