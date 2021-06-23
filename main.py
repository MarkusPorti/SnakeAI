import gym
# noinspection PyUnresolvedReferences
import gym_snake_rl

from agent import Agent
from helper import plot
from model import DQN
import torch
import torch.nn as nn        # Pytorch neural network package
import torch.optim as optim  # Pytorch optimization package

device = torch.device("cpu")


def train(epochs=10000):
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    game = gym.make('BasicSnake-rgb-16-v0')
    agent = Agent(game)

    for frame_idx in range(epochs):
        # get old state
        state_old = game.reset()

        while True:
            # get move
            final_move = agent.get_action(state_old)  # 0, 1, 2 oder 3

            # perform move and get new state
            state_new, reward, done, info = game.step(final_move)

            # train short memory
            agent.train_short_memory(state_old, final_move, reward, state_new, done)

            # remember
            agent.remember(state_old, final_move, reward, state_new, done)

            if done:
                # train long memory, plot result
                agent.n_games += 1
                agent.train_long_memory()

                if info > record:
                    record = info
                    agent.model.save()

                print('Game', agent.n_games, 'Score', info, 'Record:', record)

                plot_scores.append(info)
                total_score += info
                mean_score = total_score / agent.n_games
                plot_mean_scores.append(mean_score)
                plot(plot_scores, plot_mean_scores)
                break


def play_manual(rounds=1):
    env = gym.make('BasicSnake-rgb-16-v0')
    for counter_games in range(rounds):
        env.reset()

        next_move = 1
        # run one game till snake dies
        while env.running:
            env.clock.tick(5)

            env.step(next_move)

    quit()


def test_net():
    env = gym.make('BasicSnake-rgb-16-v0')
    net = DQN(env.observation_space.shape, env.action_space.n).to(device)
    print(net)


if __name__ == "__main__":
    # test_net()
    train()
    # agent = Agent()
    # # play_manual()
    # agent.train()
