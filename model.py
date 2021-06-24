import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 16, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(16, 64, kernel_size=(2, 2)),
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

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, states, actions, rewards, next_states, dones):
        states = torch.tensor(states, dtype=torch.float)
        next_states = torch.tensor(next_states, dtype=torch.float)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float)
        dones = torch.tensor(dones, dtype=torch.bool)
        # (n, x)

        # 1: predicted Q values with current states
        preds = self.model(states)

        if dones.dim() == 0:
            # keine Liste aus items, sondern nur 1 Item gegeben!
            targets = preds.clone()
            q_new = rewards.item()
            if not dones.item():
                q_new = rewards.item() + self.gamma * torch.max(self.model(next_states))

            targets[0][torch.argmax(actions)] = q_new
        else:
            targets = preds.clone()
            for idx in range(len(dones)):
                q_new = rewards[idx]
                if not dones[idx]:
                    q_new = rewards[idx] + self.gamma * torch.max(self.model(next_states[idx]))

                targets[idx][torch.argmax(actions[idx])] = q_new

        # 2: q_new = r + y * max(next_predicted Q value) -> only do this if not done
        # preds.clone()
        # preds[argmax(action)] = q_new
        self.optimizer.zero_grad()
        loss = self.criterion(targets, preds)
        loss.backward()

        self.optimizer.step()

