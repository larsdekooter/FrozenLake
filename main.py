import gymnasium as gym
import torch.nn as nn
import torch
from collections import deque
import torch.optim as optim
import numpy as np
import random


env = gym.make(
    "FrozenLake-v1", desc=None, map_name="4x4", is_slippery=False, render_mode="human"
)


state, info = env.reset()

BATCH_SIZE = 64


class DQN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stack = nn.Sequential(nn.Linear(1, 64), nn.ReLU(), nn.Linear(64, 4))

    def forward(self, x):
        return self.stack(x)


class Network:
    def __init__(self) -> None:
        self.model = DQN()
        self.targetModel = DQN()
        self.targetModel.load_state_dict(self.model.state_dict())
        self.steps = 0
        self.memory = deque([], maxlen=100_000)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-4)
        self.criterion = nn.SmoothL1Loss()

    def getAction(self, state):
        epsilon = 0.1 + 0.9 * np.exp(-1e-5 * self.steps)
        self.steps += 1
        if np.random.random() < epsilon:
            return env.action_space.sample()
        else:
            with torch.no_grad():
                return self.model(torch.tensor([state], dtype=torch.float)).argmax().item()

    def train(self, state, nextState, action, reward, done):
        self.memory.append(
            (
                state,
                nextState,
                action,
                reward,
                done,
            )
        )
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        states, nextStates, actions, rewards, dones = zip(*batch)

        states = torch.tensor(states, dtype=torch.float32).unsqueeze(1)
        nextStates = torch.tensor(nextStates, dtype=torch.float32).unsqueeze(1)
        actions = torch.tensor(actions)
        rewards = torch.tensor(rewards)
        dones = torch.tensor(dones, dtype=torch.int64)

        qValues = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        nextQValues = self.targetModel(nextStates).max(1)[0]
        targetQValues = rewards + (0.99 * nextQValues * (1 - dones))

        loss = self.criterion(qValues, targetQValues.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def updateTarget(self):
        self.targetModel.load_state_dict(self.model.state_dict())


network = Network()
ngames = 0
while True:

    action = network.getAction(state)
    nextState, reward, terminated, truncated, info = env.step(action)

    network.train(state, nextState, action, reward, terminated or truncated)

    state = nextState

    if truncated or terminated:
        ngames += 1
        print(ngames, network.steps)
        if ngames % 10 == 0:
            network.updateTarget()
        state, info = env.reset()
