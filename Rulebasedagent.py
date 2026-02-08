# ===================================================================================
# === Rule-Based Agent (State-Based, No env.agent_pos) ===
# ===================================================================================
print("\n Training Rule-Based Agent...")
import numpy as np
from env import SmashyRoadEnv


# Create environment
env = SmashyRoadEnv(grid_size=10)

# Q-table for learning fallback
Q = np.zeros((10, 10, 10, 10, 4))
alpha = 0.1
gamma = 0.99

def get_state_tuple(state):
    ax, ay, px, py = [int(round(x)) for x in state]
    ax = np.clip(ax, 0, 9)
    ay = np.clip(ay, 0, 9)
    px = np.clip(px, 0, 9)
    py = np.clip(py, 0, 9)
    return (ax, ay, px, py)

rb_rewards = []
rb_wins = []

print("Training Rule-Based Agent...")
for ep in range(5000):
    state = env.reset()
    done = False
    total_reward = 0
    trajectory = []

    while not done:
        # Extract positions from state tuple
        ax, ay, px, py = [int(round(x)) for x in state]
        ax = np.clip(ax, 0, 9)
        ay = np.clip(ay, 0, 9)
        px = np.clip(px, 0, 9)
        py = np.clip(py, 0, 9)

        # Police proximity check
        if abs(ax - px) <= 1 and abs(ay - py) <= 1:
            if ax < px and ax < 9:
                action = 1  # Down
            elif ay < py and ay < 9:
                action = 3  # Right
            elif ax > 0:
                action = 0  # Up
            elif ay > 0:
                action = 2  # Left
            else:
                action = 1
        else:
            # Move toward goal (9,9)
            if ax < 9:
                action = 1  # Down
            elif ay < 9:
                action = 3  # Right
            elif ax > 0:
                action = 0  # Up
            elif ay > 0:
                action = 2  # Left
            else:
                action = 0

        next_state, reward, done = env.step(action)
        s_tuple = get_state_tuple(state)
        trajectory.append((s_tuple, action, reward))
        state = next_state
        total_reward += reward

    # Monte Carlo update
    G = 0
    for s, a, r in reversed(trajectory):
        G = r + gamma * G
        Q[s][a] = (1 - alpha) * Q[s][a] + alpha * G

    won = 1.0 if env.won else 0.0
    rb_rewards.append(total_reward)
    rb_wins.append(won)

    if ep % 500 == 0:
        win_rate = np.mean(rb_wins[-100:]) if len(rb_wins) >= 100 else 0
        print(f"Episode {ep}, Rule-Based Win Rate: {win_rate:.2f}")

# Final win rate
rule_based_win_rate = np.mean(rb_wins[-100:])
print(f"Rule-Based Agent Final Win Rate: {rule_based_win_rate:.2f}")

# Save results
np.save("results/models/rb_q_table.npy", Q)
np.save("results/logs/rb_rewards.npy", rb_rewards)
np.save("results/logs/rb_wins.npy", rb_wins)