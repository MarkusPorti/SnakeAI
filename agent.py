import random
from collections import deque

import numpy as np
import torch

from model import DQN, QTrainer

DEVICE = torch.device("cpu")
MAX_MEMORY = 2_048
BATCH_SIZE = 128
LR = 0.001


class Agent:
    def __init__(self, gym_env):
        self.gamma = 0.9
        self.epsilon = 1.
        self.eps_decacy = .999955
        self.eps_min = .02

        self.n_games = 0
        self.env = gym_env
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = DQN(self.env.observation_space.shape, self.env.action_space.n).to(DEVICE)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        print(self.model)

    def remember(self, state, action, reward, next_state, done):
        # popleft if MAX_MEMORY is reached
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        # for state, action, reward, next_state, done in mini_sample:
        #     self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = max(self.epsilon * self.eps_decacy, self.eps_min)
        if np.random.random() < self.epsilon:
            final_move = self.env.action_space.sample()
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move = move

        return final_move
