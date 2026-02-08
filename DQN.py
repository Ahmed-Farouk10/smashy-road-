# policy_gradient.py → now implements DQN instead
import numpy as np
from env import SmashyRoadEnv
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque

# Device
device = torch.device("cpu")

# Deep Q-Network
class DQN(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, output_dim=4):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

def policy_gradient(env, episodes=10000, alpha=0.001, gamma=0.99, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=5000):
    # DQN setup
    model = DQN().to(device)
    target = DQN().to(device)
    target.load_state_dict(model.state_dict())
    optimizer = optim.Adam(model.parameters(), lr=alpha)
    replay_buffer = deque(maxlen=10000)

    def select_action(state, epsilon):
        if random.random() < epsilon:
            return random.randint(0, 3)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = model(state_tensor)
            return q_values.argmax().item()

    def optimize():
        if len(replay_buffer) < 1000:
            return
        batch = random.sample(replay_buffer, 32)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.BoolTensor(dones).to(device)

        current_q = model(states).gather(1, actions.unsqueeze(1)).squeeze()
        next_q = target(next_states).max(1)[0]
        expected_q = rewards + gamma * next_q * (~dones)

        loss = F.mse_loss(current_q, expected_q)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Logging
    rewards = []
    wins = []

    print("Training DQN (Deep Q-Network)...")
    for ep in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False

        epsilon = epsilon_end + (epsilon_start - epsilon_end) * np.exp(-ep / epsilon_decay)

        while not done:
            action = select_action(state, epsilon)
            next_state, reward, done = env.step(action)

            replay_buffer.append((state, action, reward, next_state, done))
            optimize()

            state = next_state
            total_reward += reward

        won = 1.0 if env.won else 0.0
        rewards.append(total_reward)
        wins.append(won)

        if ep % 500 == 0:
            win_rate = np.mean(wins[-100:]) if len(wins) >= 100 else 0
            avg_reward = np.mean(rewards[-100:])
            print(f"Episode {ep}, Win Rate: {win_rate:.2f}, Avg Reward: {avg_reward:.2f}")

        # Update target network
        if ep % 1000 == 0:
            target.load_state_dict(model.state_dict())

    # Return dummy weights for compatibility
    W = model.net[0].weight.detach().numpy()
    b = model.net[0].bias.detach().numpy()
    return W, b, rewards, wins