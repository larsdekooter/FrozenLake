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
        self.networkSteps = 0
        self.randomSteps = 0

    def getAction(self, state):
        epsilon = 0.1 + 0.9 * np.exp(-1e-5 * self.steps)
        self.steps += 1
        if np.random.random() < epsilon:
            self.randomSteps += 1
            return env.action_space.sample()
        else:
            self.networkSteps += 1
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

        pred = self.model(states)

        target = pred.clone()
        for idx in range(len(dones)):
            Qnew = rewards[idx]
            if not dones[idx]:
                Qnew = rewards[idx] + 0.99 * torch.max(self.model(nextStates[idx]))
            
            target[idx][torch.argmax(actions[idx]).item()] = Qnew
        
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
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
        networkSteps, randomSteps = network.networkSteps, network.randomSteps
        network.networkSteps = 0
        network.randomSteps = 0
        ngames += 1
        print(ngames, network.steps, round(networkSteps / (networkSteps + randomSteps) * 100.0, 2))
        if ngames % 10 == 0:
            network.updateTarget()
        state, info = env.reset()
