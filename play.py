# play.py
import numpy as np
from env import SmashyRoadEnv
import pygame

Q = np.load("results/models/q_table.npy")
env = SmashyRoadEnv(grid_size=10)

for episode in range(5):
    state = env.reset()
    done = False
    total_reward = 0
    steps = 0
    print(f"\nPlaying Episode {episode + 1}...")

    while not done:
        ax, ay, px, py = np.clip([int(x) for x in state], 0, 9)
        action = np.argmax(Q[ax, ay, px, py])
        state, reward, done = env.step(action)
        total_reward += reward
        steps += 1
        env.render()
        if env.won:
            print(f"🎉 WON! Reward: {total_reward}, Steps: {steps}")
        elif env.game_over:
            print(f"💥 CAUGHT! Reward: {total_reward}, Steps: {steps}")

pygame.quit()
print("Demo ended.")