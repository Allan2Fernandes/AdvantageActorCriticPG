import sys
import gymnasium as gym
import torch
import numpy as np
import random
from collections import deque
from A2CAgent import A2CAgent

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Create environment
env = gym.make('InvertedDoublePendulum-v4', render_mode='human')

obs_space_dims = env.observation_space.shape[0]
action_space_dims = env.action_space.shape[0]

# Constants
num_episodes = sys.maxsize
max_timesteps = 100000
statistics_window_size = 50

rewards_moving_window = deque(maxlen=statistics_window_size)

agent = A2CAgent(action_space_size=action_space_dims, state_space_size=obs_space_dims, device=device)

def preprocess_state(state):
    state = torch.tensor(state, dtype=torch.float32)
    state = torch.unsqueeze(state, dim=0)
    state = state.to(device)
    return state

for episode in range(num_episodes):
    state, _ = env.reset()
    state = preprocess_state(state)
    episode_rewards = []
    for t in range(max_timesteps):
        action, action_log_prob = agent.act(state, device)
        action = action.to('cpu')
        next_state, reward, done, truncated, info = env.step(action)
        episode_rewards.append(reward)
        # In either case, the next state doesn't exist, so it needs to factor into the calculation for advantage
        done = done or truncated
        next_state = preprocess_state(next_state)

        # Send the reward and next state back for training
        agent.train_both_networks(reward=reward, log_prob=action_log_prob, state=state, next_state=next_state, done=done)

        state = next_state

        if done:
            rewards_moving_window.append(np.sum(episode_rewards))
            print("Episode:", episode, "Average Reward:", np.mean(rewards_moving_window))
            #print("Episode {0} reward = {1}".format(episode, np.mean(episode_rewards)))
            break