# SmashyRoad: Reinforcement Learning Police Chase Game

A comprehensive reinforcement learning project implementing and comparing multiple RL algorithms in a 2D police chase game environment.

## 🎮 Game Overview

**SmashyRoad** is a grid-based game where an agent must navigate from the starting position (2,2) to the goal (9,9) while:
- Avoiding a greedy police officer that chases the agent
- Managing step penalties and collision avoidance

### Game Mechanics
- **Grid Size**: 10×10
- **Agent Start**: (2, 2)
- **Police Start**: (7, 7)
- **Goal Position**: (9, 9)
- **Fuel Bonus**: (5, 5) - worth +5 reward
- **Rewards**:
  - +100 for reaching goal
  - -100 for being caught
  - +5 for collecting fuel
  - -1 per step

## Firstly Review the report pdf 

## 🤖 Implemented RL Algorithms

### 1. **Q-Learning**
- Value-based, off-policy learning algorithm
- Uses epsilon-greedy exploration strategy
- **Results**: 62% win rate on 300 test episodes

### 2. **Value Iteration**
- Dynamic programming approach
- Guaranteed optimal policy
- Computes value function for all states
- **Results**: 100% win rate on 300 test episodes ⭐

### 3. **Hybrid MCTS (Monte Carlo Tree Search)**
- Combines tree search with value iteration guidance
- Uses UCB1 (Upper Confidence Bound) for node selection
- Enhanced exploration with value-based policy priors
- **Results**: 65% win rate on 100 test episodes

### 4. **Rule-Based Agent**
- State-based heuristic approach
- Intelligent navigation toward goal
- Collision avoidance logic

## 📊 Performance Summary

| Model | Win Rate (Last 100) |
|-------|:------------------:|
| Q-Learning | 62% |
| Value Iteration | **100%** ✓ |
| Hybrid MCTS | 65% |

## 📁 Project Structure

```
.
├── env.py                      # Core game environment (SmashyRoadEnv)
├── train.py                    # Training script for all agents
├── q_learning.py               # Q-Learning implementation
├── value_iteration.py          # Dynamic programming solver
├── Montecarlo.py               # Pure MCTS implementation
├── Montecarlohybird.py         # Hybrid MCTS implementation
├── policy_gradient.py          # Policy gradient methods
├── Rulebasedagent.py           # Rule-based heuristic agent
├── play.py                     # Play with trained Q-Learning agent
├── play_agents.py              # Demo all trained agents
├── play_vi.py                  # Play with Value Iteration policy
├── test_suite.py               # Unit tests
├── test_policy.py              # Policy evaluation utilities
├── utils.py                    # Helper functions
├── requirements.txt            # Python dependencies
├── results/                    # Trained models and logs
│   ├── models/
│   │   ├── q_table.npy         # Trained Q-table
│   │   └── value_iteration_policy.json
│   ├── logs/
│   │   ├── ql_rewards.npy
│   │   ├── ql_wins.npy
│   │   ├── mcts_rewards.npy
│   │   └── mcts_wins.npy
│   └── reports/
│       └── summary.txt         # Training summary
└── README.md
```

## 🚀 Quick Start

### Prerequisites
- Python 3.7+
- pip or conda

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/smashyroad-rl.git
   cd smashyroad-rl
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train all agents** (optional - pre-trained models included)
   ```bash
   python train.py
   ```

### Usage

#### Play with Q-Learning Agent
```bash
python play.py
```

#### Play with Value Iteration Agent
```bash
python play_vi.py
```

#### Demo All Agents
```bash
python play_agents.py
```

#### Run Tests
```bash
python test_suite.py
```

## 📋 Requirements

- **pygame** ≥ 2.0.0 - Game rendering and visualization
- **numpy** ≥ 1.20.0 - Numerical computing
- **matplotlib** ≥ 3.3.0 - Data visualization

See `requirements.txt` for exact versions.

## 🔍 Key Files Explained

### `env.py`
Defines the `SmashyRoadEnv` class - the core game environment with:
- State representation (agent pos, police pos)
- Action space (up, down, left, right)
- Reward function
- Pygame rendering

### `train.py`
Main training orchestration script:
- Trains Q-Learning on 30,000 episodes
- Trains Value Iteration (one-shot)
- Trains Hybrid MCTS on 1,000 episodes
- Saves models and generates summary report

### `q_learning.py`
Q-Learning implementation with:
- Epsilon-greedy exploration
- Learning rate (α) and discount factor (γ)
- Experience replay-like updates

### `value_iteration.py`
Dynamic programming solver that:
- Generates all 10,000 possible states
- Iteratively improves value estimates
- Extracts optimal policy from value function

### `play_agents.py`
Interactive demo showing all agents in action with:
- Lightweight environment for fast simulation
- MCTS node with UCB1 exploration
- Side-by-side comparison

## 💡 Algorithm Comparison

| Feature | Q-Learning | Value Iteration | Hybrid MCTS |
|---------|:----------:|:---------------:|:-----------:|
| **Learning Type** | Model-free | Model-based | Model-free |
| **Exploration** | ε-greedy | N/A (DP) | UCB1 |
| **Optimality** | Approximate | ✓ Guaranteed | Approximate |
| **Scalability** | Good | Limited | Good |
| **Win Rate** | 62% | 100% | 65% |



## 📚 Learning Resources

### Reinforcement Learning Concepts
- [OpenAI Gym Documentation](https://gym.openai.com/)
- [Sutton & Barto RL Book](http://incompleteideas.net/book/the-book-2nd.html)
- [Deep RL Hands-On](https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On)

### MCTS Resources
- [A Survey of Monte Carlo Tree Search Methods](https://ieee-explore.ieee.org/document/6564199)
- [Upper Confidence Bounds Applied to Trees](https://link.springer.com/article/10.1007/s10994-011-5258-6)

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👨‍💻 Author

Created as an Applied Machine Learning (AML) project exploring practical implementations of classic and advanced RL algorithms.

## Demo link : https://drive.google.com/file/d/1r3V9bnvlBJsNOPHnavaQ1z0W1lW28Gdc/view?usp=sharing
