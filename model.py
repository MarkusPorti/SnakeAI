import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 8, (2, 2), padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 16, (2, 2), padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, (2, 2), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 32),
            nn.ReLU(),
            nn.Linear(32, n_actions)
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
        from agent import DEVICE
        states = torch.tensor(states, dtype=torch.float).to(DEVICE)
        actions = torch.tensor(actions, dtype=torch.float).to(DEVICE)
        rewards = torch.tensor(rewards, dtype=torch.long).to(DEVICE)
        next_states = torch.tensor(next_states, dtype=torch.float).to(DEVICE)
        # (n, x)

        if len(states.shape) == 3:
            # (1, x)
            states = torch.unsqueeze(states, 0).to(DEVICE)
            actions = torch.unsqueeze(actions, 0).to(DEVICE)
            rewards = torch.unsqueeze(rewards, 0).to(DEVICE)
            next_states = torch.unsqueeze(next_states, 0).to(DEVICE)
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
