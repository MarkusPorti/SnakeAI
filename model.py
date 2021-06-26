import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(1, 1)),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        if len(x.shape) < 4:
            x = x[None, ...]
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)

    def save(self, max_score, file_name='model'):
        model_folder_path = './models'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name + str(max_score) + '.pth')
        torch.save(self.state_dict(), file_name)


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.loss = nn.MSELoss()

    def train_step(self, states, actions, rewards, next_states, dones):
        states = torch.tensor(states, dtype=torch.float)
        actions = torch.tensor(actions, dtype=torch.float)
        rewards = torch.tensor(rewards, dtype=torch.long)
        next_states = torch.tensor(next_states, dtype=torch.float)
        # (n, x)

        if len(states.shape) == 3:
            # (1, x)
            states = torch.unsqueeze(states, 0)
            actions = torch.unsqueeze(actions, 0)
            rewards = torch.unsqueeze(rewards, 0)
            next_states = torch.unsqueeze(next_states, 0)
            dones = (dones,)

        # 1: predicted Q values with current state
        predictions = self.model(states)

        # 2: Q_new = r + y * max(next_predicted Q value)
        targets = predictions.clone()
        for idx in range(len(dones)):
            q_new = rewards[idx]
            if not dones[idx]:
                q_new = rewards[idx] + self.gamma * torch.max(self.model(next_states[idx]))
            targets[idx][torch.argmax(actions[idx])] = q_new

        self.optimizer.zero_grad()
        loss = self.loss(targets, predictions)
        loss.backward()
        self.optimizer.step()
