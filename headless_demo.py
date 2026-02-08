# headless_demo.py
from env import SmashyRoadEnv
import time
import numpy as np

print("Starting Headless Demo: Police Catcher Game")
print("Running without Pygame display...")

# Create environment (disable rendering)
env = SmashyRoadEnv(grid_size=10)

def mock_render(self):
    """Mock render function that does nothing"""
    return True

# Monkey-patch render to avoid Pygame window
env.render = lambda: True

# Run episodes
num_episodes = 5
for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0
    steps = 0
    done = False

    print(f"\nEpisode {episode + 1} started...")

    while not done:
        action = env.action_space_sample()  # Random action
        state, reward, done = env.step(action)
        total_reward += reward
        steps += 1

        # Optional: print every few steps
        if steps % 20 == 0:
            print(f"  Step {steps}, Reward: {total_reward}, Agent: {env.agent_pos}, Police: {env.police_pos}")

    print(f"Episode {episode + 1} ended. Steps: {steps}, Total Reward: {total_reward}, Won: {env.won}")

print(" Headless demo completed.")