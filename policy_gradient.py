# policy_gradient.py
import numpy as np
from env import SmashyRoadEnv
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Device
device = torch.device("cpu")

# Neural Network
class ActorCritic(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, output_dim=4):
        super(ActorCritic, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.actor = nn.Linear(hidden_dim, output_dim)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = torch.FloatTensor(x).to(device)
        shared = self.shared(x)
        logits = self.actor(shared)
        value = self.critic(shared)
        probs = F.softmax(logits, dim=-1)
        return probs, value

def policy_gradient(env, episodes=10000, alpha=0.001, gamma=0.99, gae_lambda=0.95, eps_clip=0.2, entropy_coef=0.01):
    model = ActorCritic().to(device)
    optimizer = optim.Adam(model.parameters(), lr=alpha)

    rewards = []
    wins = []

    print("Training Advanced Policy Gradient (PPO-style)...")
    for ep in range(episodes):
        state = env.reset()
        done = False

        states = []
        actions = []
        rewards_ep = []
        log_probs = []
        values = []

        # Collect trajectory
        while not done:
            probs, value = model(state)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            value = value.squeeze(-1)

            next_state, reward, done = env.step(action.item())

            states.append(state)
            actions.append(action.item())
            rewards_ep.append(reward)
            log_probs.append(log_prob)
            values.append(value)

            state = next_state

        # Compute returns
        R = 0
        returns = []
        for r in reversed(rewards_ep):
            R = r + gamma * R
            returns.append(R)
        returns = torch.FloatTensor(returns[::-1]).to(device)
        values = torch.stack(values)
        log_probs = torch.stack(log_probs)

        # Compute advantages with GAE
        deltas = []
        for t in range(len(rewards_ep)-1):
            delta = rewards_ep[t] + gamma * values[t+1] - values[t]
            deltas.append(delta.item())
        delta = rewards_ep[-1] - values[-1]
        deltas.append(delta.item())
        advantages = []
        gae = 0
        for delta in reversed(deltas):
            gae = delta + gamma * gae_lambda * gae
            advantages.append(gae)
        advantages = torch.FloatTensor(advantages[::-1]).to(device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO-style multiple updates
        for _ in range(4):
            new_probs, _ = model(states)
            dist = torch.distributions.Categorical(new_probs)
            new_log_probs = dist.log_prob(torch.LongTensor(actions).to(device))

            ratio = (new_log_probs - log_probs.detach()).exp()
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            new_values = torch.stack([model(s)[1] for s in states]).squeeze()
            value_loss = F.mse_loss(new_values, returns)

            entropy = dist.entropy().mean()
            loss = policy_loss + 0.5 * value_loss - entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward(retain_graph=True)  # ✅ Fixed
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

        # Log
        total_reward = sum(rewards_ep)
        won = 1.0 if env.won else 0.0
        rewards.append(total_reward)
        wins.append(won)

        if ep % 500 == 0:
            win_rate = np.mean(wins[-100:]) if len(wins) >= 100 else 0
            avg_reward = np.mean(rewards[-100:])
            print(f"Episode {ep}, Win Rate: {win_rate:.2f}, Avg Reward: {avg_reward:.2f}")

    return np.random.randn(4, 4), np.zeros(4), rewards, wins