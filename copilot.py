import gym
import numpy as np
import tqdm

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("eps")
args = parser.parse_args()

# Initialize the FrozenLake environment
env = gym.make('FrozenLake-v1', is_slippery=False)

# Set parameters
learning_rate = 0.8
discount_factor = 0.95
num_episodes = int(args.eps)
steps = 0

# Initialize Q-table with zeros
Q = np.zeros([env.observation_space.n, env.action_space.n])

def choose_action(state, ep=True):
    if ep:
        epsilon = 0.1 + 0.99 * np.exp(-1e-2*steps)
        if np.random.random() < epsilon:
            return env.action_space.sample()
        else:
            return np.argmax(Q[state, :])
    else:
        return np.argmax(Q[state, :])


for i in tqdm.tqdm(range(num_episodes)):
    done = False
    state, _ = env.reset()
    while not done:
        action = choose_action(state)
        newState, reward, terminated, truncated, _ = env.step(action)
        steps +=1
        Q[state, action] = Q[state, action] + learning_rate * (reward + discount_factor * np.max(Q[newState]) - Q[state, action])
        state = newState
        done = terminated or truncated




done = False
env.close()
env = gym.make('FrozenLake-v1', is_slippery=False, render_mode="human")
state, _ = env.reset()
while not done:
    action = choose_action(state, False)
    state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated