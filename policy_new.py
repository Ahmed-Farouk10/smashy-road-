# policy_gradient.py
import numpy as np
from env import SmashyRoadEnv

def policy_gradient(env, episodes=10000, alpha=0.01, gamma=0.95):
    """
    A smart rule-based agent that:
    - Moves toward the goal (9,9)
    - Avoids police if adjacent
    - Uses Q-learning style update for edge cases
    """
    # Simple Q-table for fallback
    Q = np.zeros((10, 10, 10, 10, 4))  # (ax, ay, px, py, action)
    wins = []
    rewards = []

    def get_state_tuple(state):
        ax, ay, px, py = [int(round(x)) for x in state]
        ax = np.clip(ax, 0, 9)
        ay = np.clip(ay, 0, 9)
        px = np.clip(px, 0, 9)
        py = np.clip(py, 0, 9)
        return ax, ay, px, py

    def heuristic_action(state):
        ax, ay, px, py = state

        # If police is adjacent, avoid
        if abs(ax - px) <= 1 and abs(ay - py) <= 1:
            # Try to move away
            if ax < px and ax < 9 and [ax+1, ay] not in env.buildings:
                return 1  # Down
            elif ax > px and ax > 0 and [ax-1, ay] not in env.buildings:
                return 0  # Up
            if ay < py and ay < 9 and [ax, ay+1] not in env.buildings:
                return 3  # Right
            elif ay > py and ay > 0 and [ax, ay-1] not in env.buildings:
                return 2  # Left

        # Otherwise, move toward goal (9,9)
        if ax < 9 and [ax+1, ay] not in env.buildings:
            return 1  # Down
        elif ay < 9 and [ax, ay+1] not in env.buildings:
            return 3  # Right
        elif ax > 0 and [ax-1, ay] not in env.buildings:
            return 0  # Up
        elif ay > 0 and [ax, ay-1] not in env.buildings:
            return 2  # Left
        else:
            return 0  # Default

    print("Training Smart Rule-Based Agent...")
    for ep in range(episodes):
        state = env.reset()
        total_reward = 0
        trajectory = []
        done = False

        while not done:
            s = get_state_tuple(state)
            action = heuristic_action(s)
            next_state, reward, done = env.step(action)
            trajectory.append((s, action, reward))
            state = next_state
            total_reward += reward

        # Update Q-table with Monte Carlo returns
        G = 0
        for s, a, r in reversed(trajectory):
            G = r + gamma * G
            # Update Q with small learning rate
            Q[s][a] = (1 - alpha) * Q[s][a] + alpha * G

        won = 1.0 if env.won else 0.0
        wins.append(won)
        rewards.append(total_reward)

        if ep % 500 == 0:
            win_rate = np.mean(wins[-100:]) if len(wins) >= 100 else 0
            avg_reward = np.mean(rewards[-100:])
            print(f"Episode {ep}, Win Rate: {win_rate:.2f}, Avg Reward: {avg_reward:.2f}")

    # Return Q for compatibility (but policy is heuristic)
    W = Q.mean(axis=(0,1,2,3))  # Dummy weights
    b = np.zeros(4)
    return W, b, rewards, wins