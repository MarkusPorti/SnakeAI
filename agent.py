import time

import numpy as np
import pygame
from keras import layers
from keras.layers.core import Dense, Flatten
from keras.models import Sequential, Model
from keras.optimizers import RMSprop

from experience_replay import ExperienceReplay, Experience
from game import Game


class Agent:
    def __init__(self):
        self.gamma = 0.99
        self.batch_size = 100
        self.learning_rate = 1e-4
        self.eps_start = 1.
        self.eps_decacy = .999955
        self.eps_min = .02

        self.exp_buffer = ExperienceReplay(capacity=2500)
        self.env = Game(12, 8, gui=True)
        self.model: Model = self.__create_model()
        self.losses = []
        self._reset()

    def _reset(self):
        self.state, _, _ = self.env.reset_game()
        self.total_reward = 0.0

    def __create_model(self):
        _model = Sequential()
        _model.add(layers.Convolution2D(16, (3, 3), activation='relu',
                                        input_shape=self.env.get_input_shape()))
        _model.add(layers.Convolution2D(32, (1, 1), activation='relu'))
        _model.add(Flatten())
        _model.add(Dense(256, activation='relu'))
        _model.add(Dense(4))
        _model.compile(RMSprop(), 'MSE')
        return _model

    def train(self, epochs=10000):
        epsilon = self.eps_start
        total_rewards = []

        for frame_idx in range(epochs):
            # print('Simulation: ', frame_idx, '\tScore: ', max_score, end='\r')
            # if epsilon > self.eps_min:
            #     epsilon -= .9 / (epochs / 2)

            # run one game till snake dies
            while self.env.running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        quit()

                epsilon = max(epsilon * self.eps_decacy, self.eps_min)

                # total reward for this game
                reward = self.__play_step(epsilon)

                if reward is not None:
                    total_rewards.append(reward)
                    mean_reward = np.mean(total_rewards[-100:])
                    print("Epoche %d: avg. reward %.3f, (epsilon %.4f)" %
                          (frame_idx, mean_reward, epsilon))

            self._reset()

            if len(self.exp_buffer) < self.batch_size:
                continue

            # tic = time.perf_counter()
            minibatch = self.exp_buffer.sample(self.batch_size)
            states = []
            targets = []
            for state, action, reward, done, next_state in minibatch:
                target = reward
                if not done:
                    target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
                target_f = self.model.predict(state)
                target_f[0][np.argmax(action)] = target
                states.append(state.reshape(self.env.get_input_shape()))
                targets.append(target_f[0])
            self.model.train_on_batch(np.array(states), np.array(targets))
            # print("Train/Fit Model took %.4fs" % (time.perf_counter() - tic), end='\n\n')

        # save current weights of self.model
        self.model.save_weights('checkpoints/my_checkpoint')

    def __play_step(self, epsilon=.0):
        done_reward = None
        if np.random.random() < epsilon:
            action = self.env.step_random()
        else:
            prediction = self.model.predict(self.state)
            action = np.argmax(prediction[0])

        new_state, reward, is_done = self.env.step(action)
        self.total_reward += reward

        exp = Experience(self.state, action, reward, is_done, new_state)
        self.exp_buffer.append(exp)
        self.state = new_state
        if is_done:
            done_reward = self.total_reward
        return done_reward

    def play_manual(self, rounds=1):
        for counter_games in range(rounds):
            print('Game ', counter_games, '\r', end='')
            self.env.reset_game()

            final_move = 1
            # run one game till snake dies
            while self.env.running:
                self.env.clock.tick(5)

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        quit()

                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_DOWN:
                            final_move = 0
                        elif event.key == pygame.K_UP:
                            final_move = 1
                        elif event.key == pygame.K_RIGHT:
                            final_move = 2
                        elif event.key == pygame.K_LEFT:
                            final_move = 3
                        break

                self.env.step(final_move)

        pygame.quit()
        quit()
