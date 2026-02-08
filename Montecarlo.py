# mcts_agent.py
import numpy as np
from env import SmashyRoadEnv
import random
import copy

# Node for MCTS
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
        if self.visits == 0:
            return float('inf')
        return self.wins / self.visits + c * np.sqrt(np.log(self.parent.visits) / self.visits)

def simulate_random_playout(env):
    """Run a random playout from current env state"""
    temp_env = copy.deepcopy(env)
    done = False
    while not done:
        action = temp_env.action_space_sample()
        _, _, done = temp_env.step(action)
        if temp_env.won:
            return 1.0
        if temp_env.game_over:
            return 0.0
    return 0.0

def mcts_action(env, num_simulations=100):
    root = MCTSNode(state=env._get_state())

    for _ in range(num_simulations):
        node = root
        temp_env = copy.deepcopy(env)

        # Selection
        while node.untried_actions == [] and node.children:
            node = max(node.children, key=lambda n: n.ucb1())
            temp_env.step(node.action)

        # Expansion
        if node.untried_actions:
            action = random.choice(node.untried_actions)
            node.untried_actions.remove(action)
            _, _, done = temp_env.step(action)
            child_state = temp_env._get_state()
            child = MCTSNode(state=child_state, parent=node, action=action)
            node.children.append(child)
            node = child

        # Simulation
        reward = simulate_random_playout(temp_env)

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
        return env.action_space_sample()

def mcts_agent_train(env, episodes=5000):
    """Train using MCTS as policy"""
    print("Training MCTS Agent...")
    rewards = []
    wins = []

    for ep in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = mcts_action(env, num_simulations=50)
            state, reward, done = env.step(action)
            total_reward += reward

        won = 1.0 if env.won else 0.0
        rewards.append(total_reward)
        wins.append(won)

        if ep % 500 == 0:
            win_rate = np.mean(wins[-100:]) if len(wins) >= 100 else 0
            print(f"Episode {ep}, MCTS Win Rate: {win_rate:.2f}")

    final_win_rate = np.mean(wins[-100:])
    print(f"MCTS Final Win Rate: {final_win_rate:.2f}")
    return final_win_rate, rewards, wins