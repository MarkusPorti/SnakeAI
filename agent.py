import random
from collections import deque

import numpy as np
import torch

import helper
from model import DQN, QTrainer
from snake_game import SnakeGameAI

DEVICE = torch.device("cpu")
MAX_MEMORY = 100_000
BATCH_SIZE = 2000
LR = 0.0001


class Agent:
    def __init__(self):
        self.gamma = 0.7
        self.epsilon = 0.5
        self.eps_decacy = 0.00003
        self.eps_min = 0.001

        self.n_games = 0
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = DQN(20, 3).to(DEVICE)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        print(self.model)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = max(self.epsilon - self.eps_decacy, self.eps_min)
        final_move = [0, 0, 0]
        if np.random.random() < self.epsilon:
            move = random.randint(0, 2)
        else:
            state0 = torch.tensor(state, dtype=torch.float).to(DEVICE)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()

        final_move[move] = 1
        return final_move

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

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)


def train():
    plot_scores = []
    plot_mean_scores = []
    plot_short_mean_scores = []
    total_score = 0
    record_score = 0

    game = SnakeGameAI()
    agent = Agent()

    while True:
        # get old/current state
        state_old = game.get_state()

        # get move
        final_move = agent.get_action(state_old)  # [0, 0, 1]

        # perform move and get new state
        reward, done, score = game.play_step(final_move)

        state_new = game.get_state()

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            game.reset()
            agent.n_games += 1

            agent.train_long_memory()

            if score > record_score:
                record_score = score
                agent.model.save(record_score)

            print("Game", agent.n_games, "Score:", score, "Record:", record_score, "Epsilon:", round(agent.epsilon, 4))

            plot_scores.append(score)
            total_score += score
            plot_mean_scores.append(total_score / agent.n_games)
            plot_short_mean_scores.append(sum(plot_scores[-25:]) / 25)
            helper.plot(plot_scores, plot_mean_scores, plot_short_mean_scores)


if __name__ == '__main__':
    train()
