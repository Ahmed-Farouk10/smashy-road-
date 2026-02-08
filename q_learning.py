# q_learning.py
import numpy as np
from env import SmashyRoadEnv
import random

def q_learning(env, episodes=30000, alpha=0.1, gamma=0.95,
               epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.9999):
    N = env.grid_size
    Q = np.zeros((N, N, N, N, 4))
    rewards = []
    wins = []

    epsilon = epsilon_start

    for ep in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        won = False

        while not done:
            ax, ay, px, py = np.clip([int(x) for x in state], 0, N-1)

            if random.random() < epsilon:
                action = np.random.randint(0, 4)
            else:
                action = np.argmax(Q[ax, ay, px, py])

            next_state, reward, done = env.step(action)
            total_reward += reward

            nax, nay, npx, npy = np.clip([int(x) for x in next_state], 0, N-1)

            td_target = reward + gamma * np.max(Q[nax, nay, npx, npy])
            td_error = td_target - Q[ax, ay, px, py, action]
            Q[ax, ay, px, py, action] += alpha * td_error

            state = next_state
            if env.won:
                won = True

        rewards.append(total_reward)
        wins.append(1 if won else 0)

        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        if ep % 100 == 0:
            win_rate = np.mean(wins[-100:]) if len(wins) >= 100 else 0
            print(f"Episode {ep}, Epsilon: {epsilon:.3f}, Win Rate (last 100): {win_rate:.2f}")

    return Q, rewards, wins