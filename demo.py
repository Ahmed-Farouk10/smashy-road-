# demo.py
from env import SmashyRoadEnv
import time
import pygame
print(" Starting Police Catcher Game Demo...")
env = SmashyRoadEnv(grid_size=10)

state = env.reset()
running = True

print("Game started. Close window to exit.")

while running:
    action = env.action_space_sample()  # Random action
    state, reward, done = env.step(action)
    running = env.render()

    if done:
        print(f"Episode finished. Reward: {reward}, Steps: {env.steps}")
        time.sleep(1.0)
        state = env.reset()

pygame.quit()
print("Game ended.")