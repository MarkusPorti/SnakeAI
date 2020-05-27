import pygame
from random import randint
from game import Game

import numpy as np
from keras.optimizers import Adam
from keras.models import Sequential, Model
from keras import layers

import matplotlib.pyplot as plt


class Agent:
    def __init__(self):
        # TODO load model
        self.game = Game(12, 12, gui=True)
        self.model: Model = self.__create_model()
        self.losses = []

    def __create_model(self):
        _model = Sequential()
        # _model.add(layers.Conv2D(16, (3, 3), activation='relu', input_shape=(self.game.width + 2, self.game.width + 2, 1)))
        # _model.add(layers.Conv2D(32, (3, 3), activation='relu'))
        # _model.add(layers.Flatten())
        _model.add(layers.Dense(256, activation='relu'))
        _model.add(layers.Dense(256, activation='relu'))
        _model.add(layers.Dense(128, activation='relu'))
        _model.add(layers.Dense(3, activation='softmax'))
        _model.compile(Adam(0.01), "MSE")
        return _model

    def __train_model(self, game_state_old, action, reward, game_state_new):
        epochs = 10
        train_x = game_state_old[3]  # Eingabe
        train_y = self.model.predict(game_state_old[3])  # erwarteter Wert bei jeweiliger Eingabe
        if reward > 0:
            train_y[:] = np.zeros(3)
            train_y[:, action] = 1
            # train_y[0][action] = reward + 0.8 * max(train_y[0])
        else:
            train_y[:, action] = 0
            # minimum = min(train_y[0])
            # if minimum < 0:
            #     train_y[0][action] = reward + minimum
            # else:
            #     train_y[0][action] = reward

        # train_y[0][action] = reward + 0.8 * train_y[0][action]
        history = self.model.fit(train_x, train_y, epochs=epochs, verbose=0)
        self.losses = self.losses + history.history['loss']

        # target = reward + 0.9 * np.amax(self.model.predict(next_state[3])[0])
        # target_f = self.model.predict(state[3])
        # target_f[0][action] = target
        # self.model.fit(state[3], target_f, verbose=0)

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
                epsilon -= .9 / (500 / 2)

            # run one game till snake dies
            while self.game.running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        quit()

                # get old state
                state_old = self.game.get_state()

                # at the beginning more random, lately more advised actions
                if np.random.random() > epsilon:
                    # use prediction
                    prediction = self.model.predict(state_old[3])
                    final_move = np.argmax(prediction[0])
                else:
                    final_move = randint(0, 2)

                old_score = state_old[2].score
                state_new = self.game.step(final_move)
                reward = 0
                if not state_new[0]:
                    reward = -1 * len(self.game.snake.snake)
                elif state_new[2].score > old_score:
                    reward = len(self.game.snake.snake)

                # TODO: checken ob das okay ist... oder ob 0 mit gamma*x doch berechnet werden soll
                if reward != 0:
                    self.__train_model(state_old, final_move, reward, state_new)

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

        # plt.scatter(counter_plot, score_plot)
        # plt.draw()
        # # plot results
        # plt.show()

        # # plot losses
        tmp = range(0, len(self.losses))
        plt.plot(tmp, self.losses)
        plt.show()

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
