import random
from collections import deque

import numpy as np
import torch

from model import DQN, QTrainer
from snake_game import SnakeGameAI

LOG_VERSION = "Try_10_(18_324)"
DEVICE = torch.device("cpu")
MAX_MEMORY = 100_000
BATCH_SIZE = 100
LR = 0.0001


class Agent:
    def __init__(self):
        self.gamma = 0.8
        self.eps_max = 0.4
        self.eps_decacy = 0.0025
        self.eps_min = 0

        self.n_games = 0
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = DQN(18, 3).to(DEVICE)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        print(self.model)

    def get_action(self, state, epoch):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = max(self.eps_max - self.eps_decacy * epoch, self.eps_min)
        self.trainer.writer.add_scalar("Metrics/epsilon", self.epsilon, epoch)
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

    def train_long_memory(self, epoch):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones, epoch)

    def train_short_memory(self, state, action, reward, next_state, done, epoch):
        self.trainer.train_step(state, action, reward, next_state, done, epoch)


def train(epochs=500):
    scores = []
    plot_mean_scores = []
    plot_short_mean_scores = []
    total_score = 0
    record_score = 0

    game = SnakeGameAI()
    agent = Agent()

    agent.trainer.writer.add_text("Batch", "Max-Memory: %d, Batch-Size: %d" % (MAX_MEMORY, BATCH_SIZE))
    agent.trainer.writer.add_text("Params", "Gamma: %f\n"
                                            "Eps-Start: %f\n"
                                            "Eps-Decacy: %f\n"
                                            "Eps-Min: %f\n"
                                            "LR: %f"
                                  % (agent.gamma, agent.epsilon, agent.eps_decacy, agent.eps_min, LR))
    agent.trainer.writer.add_text("Model", str(agent.model))

    for epoch in range(epochs):
        done = False
        while not done:
            # get old/current state
            state_old = game.get_state()

            # get move
            final_move = agent.get_action(state_old, epoch)  # [0, 0, 1]

            # perform move and get new state
            reward, done, score = game.play_step(final_move)

            state_new = game.get_state()

            # train short memory
            agent.train_short_memory(state_old, final_move, reward, state_new, done, epoch)

            # remember
            agent.remember(state_old, final_move, reward, state_new, done)

            if done:
                game.reset()
                agent.n_games += 1

                agent.train_long_memory(epoch)

                total_score += score
                scores.append(score)
                if score > record_score:
                    record_score = score
                    agent.model.save(record_score)

                print("Game", agent.n_games, "Score:", score, "Record:", record_score, "Epsilon:",
                      round(agent.epsilon, 4))

                agent.trainer.writer.add_scalar("Game/Score", score, epoch)
                agent.trainer.writer.add_scalar("Game/Mean-Score", total_score / agent.n_games, epoch)
                agent.trainer.writer.add_scalar("Game/Short-Mean-Score", sum(scores[-25:]) / 25, epoch)
                agent.trainer.writer.flush()
                # helper.plot(scores, plot_mean_scores, plot_short_mean_scores)

    agent.trainer.writer.close()


if __name__ == '__main__':
    train()
