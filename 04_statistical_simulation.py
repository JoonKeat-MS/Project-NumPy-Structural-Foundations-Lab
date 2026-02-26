"""
File: 04_statistical_simulation.py
Author: Joon Keat Lim (JKai)
Date: 2026-02-26

Project: NumPy Structural Foundations Lab

Description:
This script performs a Monte Carlo simulation of coin flips
using NumPy vectorized computation.

It demonstrates:
- Random sampling
- Axis-based aggregation
- Empirical probability estimation
- Visualization of statistical convergence

Objectives:
1. Simulate repeated coin flips efficiently using vectorization.
2. Estimate probability of heads using large trials.
3. Observe convergence behavior.
4. Visualize statistical distribution.
"""

import numpy as np
import matplotlib.pyplot as plt

print("========== MONTE CARLO COIN SIMULATION ==========\n")

# =======================================================
# Parameters
# =======================================================

num_trials = 10000      # number of experiments
flips_per_trial = 100   # coin flips per experiment

print("Trials:", num_trials)
print("Flips per trial:", flips_per_trial)
print()

# =======================================================
# Simulation
# =======================================================

# For reproducibility
np.random.seed(42)

# 0 = Tail, 1 = Head
coins = np.random.randint(0, 2, size=(num_trials, flips_per_trial))

print("Simulation:\n", coins)
print("Simulation shape:", coins.shape)

# =======================================================
# Count heads per trial
# =======================================================

heads_count = np.sum(coins, axis = 1)

print("\nHeads count:",heads_count)
print("Heads count shape:", heads_count.shape)
print()

# =======================================================
# Empirical probability per trial
# =======================================================

probability_estimates = heads_count / flips_per_trial

print("Each estimated probability:", probability_estimates)
print("Average estimated probability:", np.mean(probability_estimates))
print()

# =======================================================
# Distribution Visualization
# =======================================================

print("Distribution of Estimated Head Probability (Histogram) is generated!!!")
plt.figure(figsize=(10,6))  
plt.hist(probability_estimates, bins=30, color="skyblue", edgecolor="black", alpha=0.7)
plt.title("Distribution of Estimated Head Probability", fontsize=14, fontweight="bold")
plt.xlabel("Estimated Probability", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.grid(axis="y", linestyle="--", alpha=0.7) 
plt.tight_layout()  
plt.show()

# =======================================================
# Law of Large Numbers Demonstration
# =======================================================

print("Law of large numbers demonstration is generated!!!")
cumulative_heads = np.cumsum(coins.flatten())
total_flips = np.arange(1, cumulative_heads.size + 1)
running_probability = cumulative_heads / total_flips
print("Running probability:", running_probability)

plt.figure(figsize=(10,6))
plt.plot(running_probability, color="blue", linewidth=1)
plt.axhline(0.5, color="red", linestyle="--", label="True Probability = 0.5", alpha=0.3)
plt.title("Convergence to True Probability (Law of Large Numbers)")
plt.xlabel("Number of Flips")
plt.ylabel("Running Probability")
plt.legend()
plt.grid(True)
plt.show()

print("========== END OF SIMULATION ==========")