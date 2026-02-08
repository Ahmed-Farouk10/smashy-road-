# utils.py
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_training():
    os.makedirs("results/plots", exist_ok=True)

    # Load data
    ql_rewards = np.load("results/logs/ql_rewards.npy")
    ql_wins = np.load("results/logs/ql_wins.npy")
    pg_rewards = np.load("results/logs/pg_rewards.npy")
    pg_wins = np.load("results/logs/pg_wins.npy")

    # Smoothed win rate
    def smooth(x, window=50):
        return np.convolve(x, np.ones(window)/window, mode='valid')

    # Plot 1: Rewards
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(smooth(ql_rewards), label="Q-Learning", alpha=0.8)
    plt.plot(smooth(pg_rewards), label="Policy Gradient", alpha=0.8)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Reward Curve")
    plt.legend()
    plt.grid(alpha=0.3)

    # Plot 2: Win Rate
    plt.subplot(1, 2, 2)
    plt.plot(smooth(ql_wins), label="Q-Learning", alpha=0.8)
    plt.plot(smooth(pg_wins), label="Policy Gradient", alpha=0.8)
    plt.xlabel("Episode")
    plt.ylabel("Win Rate (smoothed)")
    plt.title("Win Rate Over Time")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("results/plots/training_analysis.png")
    plt.show()

if __name__ == "__main__":
    plot_training()