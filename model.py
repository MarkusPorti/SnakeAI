import os

import torch
import torch.nn as nn
import torch.optim as optim


class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()

        self.linear = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, output_size)
        )

    def forward(self, x):
        return self.linear(x)

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

        if len(states.shape) == 1:
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
