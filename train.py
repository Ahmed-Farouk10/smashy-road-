# train.py
from env import SmashyRoadEnv
from q_learning import q_learning
from value_iteration import value_iteration
import numpy as np
import os
import json
import random

# Create result directories
os.makedirs("results/models", exist_ok=True)
os.makedirs("results/logs", exist_ok=True)
os.makedirs("results/reports", exist_ok=True)


# === Test Function for Value Iteration ===
def test_policy(env, policy, episodes=100):
    wins = 0
    print(f"\n--- Testing Value Iteration on {episodes} episodes ---")
    for i in range(episodes):
        state = env.reset()
        done = False
        steps = 0
        while not done:
            ax, ay, px, py = [int(round(x)) for x in state]
            ax = np.clip(ax, 0, 9)
            ay = np.clip(ay, 0, 9)
            px = np.clip(px, 0, 9)
            py = np.clip(py, 0, 9)
            s = (ax, ay, px, py)

            action = policy.get(s)
            if action is None:
                action = np.random.randint(0, 4)

            state, reward, done = env.step(action)
            steps += 1

            if env.won:
                wins += 1
                if i < 3:
                    print(f"Episode {i+1}:  WON in {steps} steps!")
            elif env.game_over and i < 3:
                if steps < 100:
                    print(f"Episode {i+1}:  CAUGHT at step {steps}")

    win_rate = wins / episodes
    print(f" Value Iteration Win Rate: {win_rate:.2f}")
    return win_rate


# === Q-Learning ===
print(" Training Q-Learning Agent...")
env_ql = SmashyRoadEnv(grid_size=10)
Q, ql_rewards, ql_wins = q_learning(env_ql, episodes=30000)
np.save("results/models/q_table.npy", Q)
np.save("results/logs/ql_rewards.npy", ql_rewards)
np.save("results/logs/ql_wins.npy", ql_wins)
print(f"Q-Learning Final Win Rate: {np.mean(ql_wins[-100:]):.2f}")


# === Value Iteration ===
print("\n Training Value Iteration...")
env_vi = SmashyRoadEnv(grid_size=10)
policy = value_iteration(env_vi)

# Save policy
with open("results/models/value_iteration_policy.json", "w") as f:
    policy_str = {str(k): v for k, v in policy.items()}
    json.dump(policy_str, f)

# Test
vi_win_rate = test_policy(env_vi, policy, episodes=300)


# ===================================================================================
# === Lightweight Environment for Fast MCTS Simulation ===
# ===================================================================================
class LightweightSmashyRoadEnv:
    def __init__(self, grid_size=10, max_steps=100):
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.agent_pos = [2, 2]
        self.police_pos = [7, 7]
        self.goal_pos = [grid_size - 1, grid_size - 1]
        self.fuel_pos = [5, 5]
        self.fuel_collected = False
        self.steps = 0
        self.game_over = False
        self.won = False
        self.buildings = [
            [3, 3], [3, 4], [4, 3],
            [1, 7], [2, 7], [1, 8],
            [6, 1], [7, 1], [6, 2]
        ]

    def reset(self):
        self.agent_pos = [2, 2]
        self.police_pos = [7, 7]
        self.fuel_collected = False
        self.steps = 0
        self.game_over = False
        self.won = False
        return self._get_state()

    def _get_state(self):
        return tuple(self.agent_pos + self.police_pos)

    def step(self, action):
        ax, ay = self.agent_pos
        px, py = self.police_pos

        # Agent movement
        new_ax, new_ay = ax, ay
        if action == 0:   # Up
            new_ax = max(0, ax - 1)
        elif action == 1: # Down
            new_ax = min(self.grid_size - 1, ax + 1)
        elif action == 2: # Left
            new_ay = max(0, ay - 1)
        elif action == 3: # Right
            new_ay = min(self.grid_size - 1, ay + 1)

        if [new_ax, new_ay] not in self.buildings:
            ax, ay = new_ax, new_ay
        self.agent_pos = [ax, ay]

        # Check win
        if self.agent_pos == self.goal_pos:
            reward = 100
            done = True
            self.won = True
            return self._get_state(), reward, done

        # Police follows
        if px < ax and [px + 1, py] not in self.buildings:
            px += 1
        elif px > ax and [px - 1, py] not in self.buildings:
            px -= 1
        if py < ay and [px, py + 1] not in self.buildings:
            py += 1
        elif py > ay and [px, py - 1] not in self.buildings:
            py -= 1
        self.police_pos = [px, py]

        self.steps += 1
        reward = -1
        done = False

        # Check police collision
        if self.agent_pos == self.police_pos:
            reward = -100
            done = True
            self.game_over = True
            return self._get_state(), reward, done

        # Fuel
        if not self.fuel_collected and self.agent_pos == self.fuel_pos:
            reward = 5
            self.fuel_collected = True

        # Max steps
        if self.steps >= self.max_steps:
            done = True

        return self._get_state(), reward, done


# ===================================================================================
# === Hybrid MCTS Agent (Value Iteration-Guided) ===
# ===================================================================================
class MCTSNode:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.wins = 0
        self.visits = 0
        self.untried_actions = [0, 1, 2, 3]  # Up, Down, Left, Right

    def ucb1(self, c=1.41):
        if self.parent is None or self.visits == 0:
            return float('inf')
        return self.wins / self.visits + c * np.sqrt(np.log(self.parent.visits) / self.visits)


def simulate_random_playout(env, max_steps=50):
    """Run a random playout with step limit"""
    temp_env = LightweightSmashyRoadEnv(grid_size=10, max_steps=max_steps)
    temp_env.agent_pos = env.agent_pos[:]
    temp_env.police_pos = env.police_pos[:]
    temp_env.fuel_collected = env.fuel_collected
    temp_env.steps = env.steps
    temp_env.game_over = env.game_over
    temp_env.won = env.won

    done = False
    steps = 0
    while not done and steps < max_steps:
        action = random.randint(0, 3)
        _, _, done = temp_env.step(action)
        if temp_env.won:
            return 1.0
        if temp_env.game_over:
            return 0.0
        steps += 1
    return 0.0  # Timeout → assume loss


def mcts_action(env, num_simulations=20, value_policy=None):
    root = MCTSNode(state=env._get_state())

    for _ in range(num_simulations):
        node = root
        temp_env = LightweightSmashyRoadEnv(grid_size=10)
        temp_env.agent_pos = env.agent_pos[:]
        temp_env.police_pos = env.police_pos[:]
        temp_env.fuel_collected = env.fuel_collected
        temp_env.steps = env.steps
        temp_env.game_over = env.game_over
        temp_env.won = env.won

        # Selection
        while node.untried_actions == [] and node.children:
            node = max(node.children, key=lambda n: n.ucb1())
            temp_env.step(node.action)

        # Expansion
        if node.untried_actions:
            # Use Value Iteration policy to guide action selection
            s = temp_env._get_state()
            ax, ay, px, py = [int(round(x)) for x in s]
            ax = np.clip(ax, 0, 9)
            ay = np.clip(ay, 0, 9)
            px = np.clip(px, 0, 9)
            py = np.clip(py, 0, 9)
            s_key = (ax, ay, px, py)

            action = value_policy.get(s_key)
            if action is None or action not in node.untried_actions:
                action = random.choice(node.untried_actions)
            else:
                action = int(action)
            node.untried_actions.remove(action)

            _, _, done = temp_env.step(action)
            child_state = temp_env._get_state()
            child = MCTSNode(state=child_state, parent=node, action=action)
            node.children.append(child)
            node = child

        # Simulation
        reward = simulate_random_playout(temp_env, max_steps=50)

        # Backpropagation
        while node:
            node.visits += 1
            node.wins += reward
            node = node.parent

    # Return best action by visit count
    if root.children:
        best_child = max(root.children, key=lambda c: c.visits)
        return best_child.action
    else:
        return random.randint(0, 3)


def train_mcts_agent(env, episodes=1000, num_simulations=20, value_policy=None):
    mcts_rewards = []
    mcts_wins = []

    print("Training Hybrid MCTS Agent...")
    for ep in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = mcts_action(env, num_simulations=num_simulations, value_policy=value_policy)
            state, reward, done = env.step(action)
            total_reward += reward

        won = 1.0 if env.won else 0.0
        mcts_rewards.append(total_reward)
        mcts_wins.append(won)

        if ep % 100 == 0:
            win_rate = np.mean(mcts_wins[-100:]) if len(mcts_wins) >= 100 else 0
            print(f"Episode {ep}, Hybrid MCTS Win Rate: {win_rate:.2f}")

    final_win_rate = np.mean(mcts_wins[-100:])
    print(f"Hybrid MCTS Final Win Rate: {final_win_rate:.2f}")

    # Save results
    np.save("results/logs/mcts_rewards.npy", mcts_rewards)
    np.save("results/logs/mcts_wins.npy", mcts_wins)

    return final_win_rate, mcts_rewards, mcts_wins


# Train Hybrid MCTS using Value Iteration policy
env_mcts = SmashyRoadEnv(grid_size=10)
mcts_win_rate, mcts_rewards, mcts_wins = train_mcts_agent(
    env_mcts, 
    episodes=1000, 
    num_simulations=20, 
    value_policy=policy
)


# === Final Summary ===
report = f"""
Training Complete!

Model               | Win Rate (Last 100)
--------------------|--------------------
Q-Learning          | {np.mean(ql_wins[-100:]):.2f}
Value Iteration     | {vi_win_rate:.2f}
Hybrid MCTS         | {mcts_win_rate:.2f}
"""

print(report)
with open("results/reports/summary.txt", "w", encoding="utf-8") as f:
    f.write(report)