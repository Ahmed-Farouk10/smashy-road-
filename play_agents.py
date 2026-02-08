# play_agents.py
import numpy as np
import json
import pygame
from env import SmashyRoadEnv

# Load Q-Table
Q = np.load("results/models/q_table.npy")

# Load Value Iteration Policy
with open("results/models/value_iteration_policy.json", "r") as f:
    policy_str = json.load(f)
    vi_policy = {eval(k): v for k, v in policy_str.items()}


# ===================================================================================
# === Lightweight Environment for Fast Simulation ===
# ===================================================================================
class LightweightSmashyRoadEnv:
    def __init__(self, grid_size=10):
        self.grid_size = grid_size
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
        if self.steps >= 100:
            done = True

        return self._get_state(), reward, done


# ===================================================================================
# === Hybrid MCTS Action Selection ===
# ===================================================================================
class MCTSNode:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.wins = 0
        self.visits = 0
        self.untried_actions = [0, 1, 2, 3]

    def ucb1(self, c=1.41):
        if self.parent is None or self.visits == 0:
            return float('inf')
        return self.wins / self.visits + c * np.sqrt(np.log(self.parent.visits) / self.visits)


def simulate_random_playout(env, max_steps=50):
    """Run a random playout with step limit"""
    temp_env = LightweightSmashyRoadEnv(grid_size=10)
    temp_env.agent_pos = env.agent_pos[:]
    temp_env.police_pos = env.police_pos[:]
    temp_env.fuel_collected = env.fuel_collected
    temp_env.steps = env.steps
    temp_env.game_over = env.game_over
    temp_env.won = env.won

    done = False
    steps = 0
    while not done and steps < max_steps:
        action = np.random.randint(0, 4)
        _, _, done = temp_env.step(action)
        if temp_env.won:
            return 1.0
        if temp_env.game_over:
            return 0.0
        steps += 1
    return 0.0  # Timeout → assume loss


def hybrid_mcts_action(env, num_simulations=20, value_policy=None):
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
            s = temp_env._get_state()
            ax, ay, px, py = [int(round(x)) for x in s]
            ax = np.clip(ax, 0, 9)
            ay = np.clip(ay, 0, 9)
            px = np.clip(px, 0, 9)
            py = np.clip(py, 0, 9)
            s_key = (ax, ay, px, py)

            action = value_policy.get(s_key)
            if action is None or action not in node.untried_actions:
                action = np.random.choice(node.untried_actions)
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
        return np.random.randint(0, 4)


# ===================================================================================
# === Run Agents ===
# ===================================================================================
def run_agent(env, name, policy=None, Q=None, agent_type=None, value_policy=None):
    print(f"\n--- Playing: {name} ---")
    for episode in range(3):
        state = env.reset()
        done = False
        steps = 0
        while not done:
            if Q is not None:  # Q-Learning
                ax, ay, px, py = np.clip([int(x) for x in state], 0, 9)
                action = np.argmax(Q[ax, ay, px, py])
            elif policy is not None:  # Value Iteration
                s = tuple(int(round(x)) for x in state)
                action = policy.get(s, np.random.randint(0, 4))
            elif agent_type == "hybrid_mcts":  # Hybrid MCTS
                action = hybrid_mcts_action(env, num_simulations=20, value_policy=value_policy)
            else:  # Random
                action = np.random.randint(0, 4)

            state, _, done = env.step(action)
            steps += 1
            env.render()
            if env.won:
                print(f"Episode {episode+1}:  Won in {steps} steps!")
            elif env.game_over:
                print(f"Episode {episode+1}:  Lost at step {steps}")
        env.clock.tick(1)


# Create environment
env = SmashyRoadEnv()

# Run all agents
print(" Demo: All Agents vs Random")
run_agent(env, "Random Agent")
run_agent(env, "Q-Learning Agent", Q=Q)
run_agent(env, "Value Iteration Agent", policy=vi_policy)
run_agent(env, "Hybrid MCTS Agent", agent_type="hybrid_mcts", value_policy=vi_policy)

pygame.quit()