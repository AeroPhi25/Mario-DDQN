#!/usr/bin/env python
# coding: utf-8

# # Dueling DQN Mario Agent – Exploration vs. Exploitation
# 
# ## Objective
# 
# This project aims to train a reinforcement learning (RL) agent using the **Dueling Deep Q-Network (Dueling DQN)** architecture to play Super Mario Bros, and to analyze how different **exploration strategies** affect performance.
# 
# ---
# 
# ##  Key Goals
# 1. Implement a Dueling DQN Mario agent using PyTorch.
# 2. Compare different exploration techniques:
#    - Epsilon decay schedules (linear vs exponential)
#    - Action space variants (aggressive vs conservative)
#    - Optional: NoisyDQN for adaptive exploration
# 3. Evaluate agent performance by:
#    - Total reward per episode
#    - Learning stability
#    - Level completion
# 
# ---
# 
# ##  Improvement (Modern Touch)
# 
# While the original Dueling DQN paper was from 2015, we enhance our implementation using ideas from:
# - **Noisy DQN**: Add learnable noise to Q-values for better exploration.
# - **Rainbow-style insights**: Integrate modular improvements incrementally.

# # Environment Setup & Preprocessing
# 
# We use the `gym-super-mario-bros` environment from OpenAI Gym, wrapped with preprocessing functions that:
# - Convert frames to grayscale
# - Resize to 84×84 resolution
# - Stack 4 consecutive frames for temporal context
# - Normalize pixel values
# 
# These are standard preprocessing techniques in deep RL, popularized by DeepMind's DQN paper.

# In[1]:


#Install Dependencies 
get_ipython().system('pip install gym==0.23.1 gym-super-mario-bros==7.3.0 nes-py==8.2.1 opencv-python torch torchvision')


# In[1]:


#Define Frame Preprocessing Wrappers
import gym
import cv2
import numpy as np
from collections import deque
from gym import spaces

class PreprocessFrame(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        self.shape = shape
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(shape[0], shape[1], 1), dtype=np.uint8
        )

    def observation(self, obs):
        return PreprocessFrame.process(obs, self.shape)

    @staticmethod
    def process(frame, shape):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, shape, interpolation=cv2.INTER_AREA)
        return np.expand_dims(frame, -1)

class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        super().__init__(env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(shp[0], shp[1], shp[2] * k), dtype=np.uint8
        )

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_observation()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_observation(), reward, done, info

    def _get_observation(self):
        return np.concatenate(list(self.frames), axis=2)


# In[2]:


#Wrap Mario Environment

import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace

def create_mario_env():
    env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = PreprocessFrame(env, shape=(84, 84))
    env = FrameStack(env, k=4)
    return env


# # Dueling Deep Q-Network (Dueling DQN)
# 
# Dueling DQN separates the Q-value into two parts:
# - **Value function V(s)**: How good is it to be in a state?
# - **Advantage function A(s, a)**: How beneficial is it to take action _a_ in that state?
# 
# These are combined as:
# 
# \[
# Q(s, a) = V(s) + \left(A(s, a) - \frac{1}{|\mathcal{A}|} \sum_{a'} A(s, a')\right)
# \]
# 
# This helps the network learn better even when the advantage of actions is similar — ideal for environments like Mario where some actions (like standing still) may not change Q-values significantly.

# In[3]:


import torch
import torch.nn as nn
import torch.nn.functional as F

class DuelingDQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DuelingDQN, self).__init__()
        
        # Convert input shape from (H, W, C) to (C, H, W)
        c, h, w = input_shape[2], input_shape[0], input_shape[1]

        self.features = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        # Compute the size of the flattened output
        with torch.no_grad():
            dummy_input = torch.zeros(1, c, h, w)
            n_flatten = self.features(dummy_input).view(1, -1).size(1)

        self.value_stream = nn.Sequential(
            nn.Linear(n_flatten, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(n_flatten, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def forward(self, x):
        x = x / 255.0  # Normalize from [0,255] to [0,1]
        x = self.features(x)
        # x = x.view(x.size(0), -1)
        x = x.reshape(x.size(0), -1)
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


# In[4]:


env = create_mario_env()
input_shape = env.observation_space.shape  # (84, 84, 4)
n_actions = env.action_space.n

model = DuelingDQN(input_shape, n_actions)
print(model)


# # Replay Buffer and Agent Logic
# 
# To stabilize training in reinforcement learning, we use an **experience replay buffer** that stores past transitions \((s, a, r, s', done)\). This allows the agent to break the correlation between sequential states and enables mini-batch updates.
# 
# The **Agent class** is responsible for:
# - Selecting actions using an ε-greedy or Noisy DQN strategy
# - Storing experiences in memory
# - Training the Dueling DQN
# - Updating the target network

# In[5]:


#Replay Buffer Class

import random
from collections import deque
import torch

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.array, zip(*batch))

        return (
            torch.tensor(state.copy(), dtype=torch.float32).permute(0, 3, 1, 2),  # (B,C,H,W)
            torch.tensor(action, dtype=torch.int64),
            torch.tensor(reward, dtype=torch.float32),
            torch.tensor(next_state.copy(), dtype=torch.float32).permute(0, 3, 1, 2),
            torch.tensor(done, dtype=torch.float32)
        )

    def __len__(self):
        return len(self.buffer)


# In[6]:


# Mario Agent Class

import torch.nn.functional as F
import torch.optim as optim

class MarioAgent:
    def __init__(self, input_shape, n_actions, device):
        self.device = device
        self.n_actions = n_actions
        self.policy_net = DuelingDQN(input_shape, n_actions).to(device)
        self.target_net = DuelingDQN(input_shape, n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-4)
        self.memory = ReplayBuffer(capacity=100_000)
        self.batch_size = 32
        self.gamma = 0.99
        self.epsilon = 1.0
        self.eps_min = 0.05
        self.eps_decay = 0.995
        self.update_target_every = 1000
        self.step_count = 0

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        state = torch.tensor(state.copy(), dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state)
        return q_values.argmax().item()

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Compute target Q-values
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            targets = rewards + self.gamma * next_q_values * (1 - dones)

        # Current Q-values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()

        # Loss and update
        loss = F.mse_loss(current_q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # # Decay epsilon
        # self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)

        # Update target network
        self.step_count += 1
        if self.step_count % self.update_target_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())


# # Training Loop
# 
# The training loop is the core engine of reinforcement learning. It:
# 1. Interacts with the environment
# 2. Stores state transitions in the replay buffer
# 3. Updates the Dueling DQN via sampled mini-batches
# 4. Tracks rewards and agent performance
# 5. Periodically saves model checkpoints
# 
# We also apply an ε-greedy policy for exploration vs exploitation during training.

# In[7]:


# Training Loop Code

import os
import time

def train_mario(num_episodes=500, save_dir="checkpoints"):
    env = create_mario_env()
    input_shape = env.observation_space.shape
    n_actions = env.action_space.n
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent = MarioAgent(input_shape, n_actions, device)
    os.makedirs(save_dir, exist_ok=True)

    rewards_per_episode = []

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0

        for step in range(10_000):  # max steps per episode
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.memory.push(state, action, reward, next_state, done)
            agent.train_step()

            state = next_state
            total_reward += reward

            if done:
                break

        rewards_per_episode.append(total_reward)
        print(f"Episode {episode + 1}/{num_episodes} | Reward: {total_reward:.2f} | Epsilon: {agent.epsilon:.3f}")
        agent.epsilon = max(agent.eps_min, agent.epsilon * agent.eps_decay)


        # Save checkpoint every 50 episodes
        if (episode + 1) % 50 == 0:
            model_path = os.path.join(save_dir, f"mario_dqn_ep{episode+1}.pth")
            torch.save(agent.policy_net.state_dict(), model_path)

    env.close()
    return rewards_per_episode


# In[8]:


#Run the Training Loop

rewards = train_mario(num_episodes=300)


# # Evaluation
# 
# After training, we evaluate our Mario agent by:
# - Loading the trained model checkpoint
# - Letting the agent play the game using its learned policy
# - Rendering gameplay to observe performance
# - Measuring reward per episode
# 
# This allows us to validate that training was effective and identify strengths or weaknesses in the learned policy.

# In[ ]:


def evaluate_trained_agent(model_path, episodes=3):
    env = create_mario_env()
    input_shape = env.observation_space.shape
    n_actions = env.action_space.n
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load trained model
    model = DuelingDQN(input_shape, n_actions).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    for ep in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            state_tensor = torch.tensor(state.copy(), dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
            with torch.no_grad():
                action = model(state_tensor).argmax().item()

            state, reward, done, _ = env.step(action)
            env.render()
            total_reward += reward
            time.sleep(0.01)  # slow down rendering for visibility

        print(f"Episode {ep + 1}: Reward = {total_reward:.2f}")

    env.close()


# In[ ]:


evaluate_trained_agent("checkpoints/mario_dqn_ep250.pth", episodes=3)


# # Visualizing Training Performance
# 
# Plotting total reward per episode helps monitor:
# - Learning progress over time
# - Agent stability and convergence
# - Impact of different exploration strategies
# 
# Smoothed reward curves reveal performance trends more clearly.

# In[9]:


import matplotlib.pyplot as plt

def plot_rewards(rewards, window=15):
    plt.figure(figsize=(10, 5))
    smoothed = [np.mean(rewards[max(0, i - window):(i + 1)]) for i in range(len(rewards))]
    plt.plot(rewards, label='Episode Reward')
    plt.plot(smoothed, label=f'{window}-Episode Moving Avg', linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Mario Agent Training Performance - Dueling DQN")
    plt.legend()
    plt.grid()
    plt.show(block=True)


# In[10]:


plot_rewards(rewards)


# In[11]:


print(f"\nAverage Reward: {np.mean(rewards):.2f}")
print(f"Best Reward: {np.max(rewards):.2f}")
print(f"Last 10 Avg Reward: {np.mean(rewards[-10:]):.2f}")


# # Noisy DQN – A Modern Exploration Strategy
# 
# Noisy DQN replaces the ε-greedy strategy with a learned distribution over Q-values by injecting noise into the network’s weights. This:
# - Enables efficient exploration without manual ε-decay tuning
# - Improves performance in sparse or deceptive reward environments
# - Is fully trainable and adaptable
# 
# Source: Fortunato et al., 2018 ([arXiv:1706.10295](https://arxiv.org/abs/1706.10295))

# In[12]:



# Define Noisy Linear Layer
# Noisy Linear Layer

import torch
import torch.nn as nn
import torch.nn.functional as F

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.FloatTensor(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer("bias_epsilon", torch.FloatTensor(out_features))

        self.sigma_init = sigma_init
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1. / self.in_features ** 0.5
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init * mu_range)

        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init * mu_range)

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)

        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))  # outer product
        self.bias_epsilon.copy_(epsilon_out)

    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def forward(self, input):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(input, weight, bias)


# In[13]:


# Define NoisyDuelingDQN


class NoisyDuelingDQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(NoisyDuelingDQN, self).__init__()
        c, h, w = input_shape  # should be (4, 84, 84)

        self.features = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU()
        )

        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            n_flatten = self.features(dummy).view(1, -1).size(1)

        self.value_stream = nn.Sequential(
            NoisyLinear(n_flatten, 512), nn.ReLU(),
            NoisyLinear(512, 1)
        )

        self.advantage_stream = nn.Sequential(
            NoisyLinear(n_flatten, 512), nn.ReLU(),
            NoisyLinear(512, n_actions)
        )

    def forward(self, x):
        x = x.float() / 255.0  # normalize
        x = self.features(x)
        x = x.view(x.size(0), -1)
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        return value + advantage - advantage.mean(dim=1, keepdim=True)

    def reset_noise(self):
        for layer in self.value_stream:
            if isinstance(layer, NoisyLinear):
                layer.reset_noise()
        for layer in self.advantage_stream:
            if isinstance(layer, NoisyLinear):
                layer.reset_noise()


# In[14]:


class MarioAgent:
    def __init__(self, input_shape, n_actions, device, use_noisy=False):
        self.device = device
        self.n_actions = n_actions
        self.use_noisy = use_noisy

        if use_noisy:
            self.policy_net = NoisyDuelingDQN(input_shape, n_actions).to(device)
            self.target_net = NoisyDuelingDQN(input_shape, n_actions).to(device)
        else:
            self.policy_net = DuelingDQN(input_shape, n_actions).to(device)
            self.target_net = DuelingDQN(input_shape, n_actions).to(device)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-4)
        self.memory = ReplayBuffer(capacity=100_000)
        self.batch_size = 32
        self.gamma = 0.99
        self.epsilon = 1.0
        self.eps_min = 0.05
        self.eps_decay = 0.995
        self.update_target_every = 1000
        self.step_count = 0

    def select_action(self, state):
        if not self.use_noisy and random.random() < self.epsilon:
            return random.randrange(self.n_actions)

        state_tensor = torch.tensor(state.copy(), dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if self.use_noisy:
                self.policy_net.reset_noise()
            q_values = self.policy_net(state_tensor)
        return q_values.argmax().item()

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        states, actions, rewards = states.to(self.device), actions.to(self.device), rewards.to(self.device)
        next_states, dones = next_states.to(self.device), dones.to(self.device)

        if self.use_noisy:
            self.policy_net.reset_noise()
            self.target_net.reset_noise()

        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            targets = rewards + self.gamma * next_q * (1 - dones)

        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()
        loss = F.mse_loss(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if not self.use_noisy:
            self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)

        self.step_count += 1
        if self.step_count % self.update_target_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())


# In[15]:


from collections import deque
import random
import torch
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        # States are already in (C, H, W) format, no need to transpose
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.array, zip(*batch))

        return (
            torch.tensor(state.copy(), dtype=torch.float32),       # (B, C, H, W)
            torch.tensor(action, dtype=torch.int64),
            torch.tensor(reward, dtype=torch.float32),
            torch.tensor(next_state.copy(), dtype=torch.float32),  # (B, C, H, W)
            torch.tensor(done, dtype=torch.float32)
        )

    def __len__(self):
        return len(self.buffer)


# In[19]:


import os
import time
import numpy as np
import torch

def train_mario_noisy(num_episodes=300, save_dir="checkpoints_noisy"):
    os.makedirs(save_dir, exist_ok=True)

    env = create_mario_env()

    # Ensure the shape is in (C, H, W) format
    raw_shape = env.observation_space.shape  # e.g., (84, 84, 4) or (4, 84, 84)
    if raw_shape[-1] == 4:  # (H, W, C)
        input_shape = (raw_shape[-1], raw_shape[0], raw_shape[1])  # (4, 84, 84)
    else:
        input_shape = raw_shape  # already (C, H, W)

    n_actions = env.action_space.n
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #  Use Noisy Dueling DQN
    agent = MarioAgent(input_shape, n_actions, device, use_noisy=True)

    rewards_per_episode = []

    for episode in range(num_episodes):
        state = env.reset()
        state = np.transpose(state, (2, 0, 1))  # Convert (H, W, C) → (C, H, W)

        total_reward = 0
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            next_state = np.transpose(next_state, (2, 0, 1))  # (C, H, W)

            agent.memory.push(state, action, reward, next_state, done)
            agent.train_step()

            state = next_state
            total_reward += reward

        rewards_per_episode.append(total_reward)
        print(f"[Noisy DQN] Episode {episode+1}/{num_episodes} ➤ Reward: {total_reward:.2f}")

        if (episode + 1) % 50 == 0:
            checkpoint_path = os.path.join(save_dir, f"noisy_mario_dqn_ep{episode+1}.pth")
            torch.save(agent.policy_net.state_dict(), checkpoint_path)

    env.close()
    return rewards_per_episode


# In[20]:


# Noisy DQN
rewards_noisy = train_mario_noisy(num_episodes=300) # can use 300


# In[ ]:


def watch_trained_noisy(agent, env, episodes=3):
    for ep in range(episodes):
        state = env.reset()
        state = np.transpose(state, (2, 0, 1))
        done = False
        total_reward = 0

        while not done:
            env.render()
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            state = np.transpose(next_state, (2, 0, 1))
            total_reward += reward

        print(f"Episode {ep+1} ➤ Total Reward: {total_reward}")
    env.close()


# In[21]:


import matplotlib.pyplot as plt

def plot_rewards(rewards_noisy, window=15):
    plt.figure(figsize=(10, 5))
    smoothed = [np.mean(rewards[max(0, i - window):(i + 1)]) for i in range(len(rewards))]
    plt.plot(rewards, label='Episode Reward')
    plt.plot(smoothed, label=f'{window}-Episode Moving Avg', linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Mario Agent Training Rewards - Noisy Dueling DQN")
    plt.legend()
    plt.grid()
    plt.show(block=True)


# In[22]:



# Plot both

plot_rewards(rewards_noisy )


# In[23]:


print(f"\nAverage Reward: {np.mean(rewards_noisy):.2f}")
print(f"Best Reward: {np.max(rewards_noisy):.2f}")
print(f"Last 10 Avg Reward: {np.mean(rewards_noisy[-10:]):.2f}")


# In[ ]:


##  Evaluate Noisy DQN Agent (with rendering)
import time
import torch
import numpy as np

def watch_noisy_agent(model_path, episodes=3):
    env = create_mario_env()
    raw_shape = env.observation_space.shape
    input_shape = (raw_shape[-1], raw_shape[0], raw_shape[1]) if raw_shape[-1] == 4 else raw_shape
    n_actions = env.action_space.n
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = NoisyDuelingDQN(input_shape, n_actions).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    for ep in range(episodes):
        state = env.reset()
        state = np.transpose(state, (2, 0, 1))  # Make it (C, H, W)
        done = False
        total_reward = 0

        while not done:
            env.render()  # ← this shows the live game
            state_tensor = torch.tensor(state.copy(), dtype=torch.float32).unsqueeze(0).to(device)
            model.reset_noise()
            with torch.no_grad():
                action = model(state_tensor).argmax().item()

            next_state, reward, done, _ = env.step(action)
            state = np.transpose(next_state, (2, 0, 1))
            total_reward += reward
            time.sleep(0.01)

        print(f"[Noisy DQN] Episode {ep+1}: Total Reward = {total_reward:.2f}")

    env.close()


# In[ ]:


watch_noisy_agent("checkpoints_noisy/noisy_mario_dqn_ep300.pth")


# In[24]:


import matplotlib.pyplot as plt
import numpy as np

def plot_comparison(rewards, rewards_noisy, window=15):
    smoothed_dqn = [np.mean(rewards[max(0, i - window):(i + 1)]) for i in range(len(rewards))]
    smoothed_noisy = [np.mean(rewards_noisy[max(0, i - window):(i + 1)]) for i in range(len(rewards_noisy))]

    plt.figure(figsize=(12, 6))
    plt.plot(rewards, alpha=0.3, label='DQN Episode Reward')
    plt.plot(smoothed_dqn, label='DQN Smoothed', linewidth=2)

    plt.plot(rewards_noisy, alpha=0.3, label='Noisy DQN Episode Reward')
    plt.plot(smoothed_noisy, label='Noisy DQN Smoothed', linewidth=2)

    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("DQN vs Noisy DQN: Training Performance")
    plt.legend()
    plt.grid(True)
    plt.show()


# In[ ]:


plot_comparison(rewards, rewards_noisy)


# In[25]:


import numpy as np
import pandas as pd

def compare_agents(dqn_rewards, noisy_rewards):
    data = {
        "Metric": ["Average Reward", "Max Reward", "Min Reward", "Std Dev"],
        "Dueling DQN": [
            np.mean(dqn_rewards),
            np.max(dqn_rewards),
            np.min(dqn_rewards),
            np.std(dqn_rewards)
        ],
        "Noisy Dueling DQN": [
            np.mean(noisy_rewards),
            np.max(noisy_rewards),
            np.min(noisy_rewards),
            np.std(noisy_rewards)
        ]
    }

    df = pd.DataFrame(data)
    return df

# Example usage (replace with your actual lists)
# dqn_rewards = [...]
# noisy_rewards = [...]
comparison_df = compare_agents(rewards, rewards_noisy)
print(comparison_df)


# In[26]:


import matplotlib.pyplot as plt
import numpy as np

# Convert total times (in seconds)
dqn_total_time_sec = 136 * 60 + 26.7
noisy_total_time_sec = 214 * 60 + 49.7
episodes = 300

# Compute average time per episode
dqn_avg = dqn_total_time_sec / episodes
noisy_avg = noisy_total_time_sec / episodes

# Generate per-episode time lists
dqn_times = [dqn_avg] * episodes
noisy_times = [noisy_avg] * episodes

# Plot training time trends
plt.figure(figsize=(10, 5))
plt.plot(dqn_times, label=f'DQN (avg: {dqn_avg:.2f}s)', color='blue')
plt.plot(noisy_times, label=f'Noisy DQN (avg: {noisy_avg:.2f}s)', color='red')
plt.xlabel("Episode")
plt.ylabel("Time per Episode (s)")
plt.title("Training Time per Episode: DQN vs Noisy DQN")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# In[ ]:




