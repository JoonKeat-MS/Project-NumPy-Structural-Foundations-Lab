# NumPy Structural Foundations Lab

## Overview

This repository is a structured experimental lab designed to build
deep intuition about how NumPy arrays behave internally.

Rather than learning NumPy as a collection of functions,
this project investigates:

- Shape mechanics
- Axis semantics
- Broadcasting rules
- Memory sharing behavior
- Vectorized statistical simulation

The goal is to develop strong structural reasoning in numerical computing —
a foundational skill for statistics, data science, and AI research.

---

# Learning Philosophy

NumPy is not just a library — it is a computational model.

Understanding it requires clarity in:

- How multidimensional arrays are structured
- How dimensions collapse during reductions
- How broadcasting expands dimensions logically
- How memory views differ from copies
- How vectorization replaces loops

This lab approaches NumPy concept-by-concept through controlled experiments.

---

# Experimental Series

This lab consists of four structured experiments:

---

## 01 — Shape & Memory Foundations

Focus:
- Array creation in 1D, 2D, 3D
- `shape` and `ndim`
- `reshape` invariance of total elements
- `transpose` axis permutation
- `flatten` vs `ravel` vs `reshape`
- Memory sharing validation using `np.shares_memory()`

Outcome:
- Understand tensor reinterpretation
- Predict shape transformations
- Distinguish view vs copy behavior

Status: Completed

---

## 02 — Axis & Reduction Semantics

Focus:
- Meaning of `axis`
- Row-wise vs column-wise aggregation
- Reduction functions (`sum`, `mean`, `max`, `min`)
- Dimensional collapse logic
- `keepdims=True`
- 3D axis experiments

Outcome:
- Predict result shapes before execution
- Understand statistical aggregation on tensors
- Develop mental shape tracing

Status: Completed

---

## 03 — Broadcasting & Dimension Expansion

Focus:
- Broadcasting rules (right-aligned comparison)
- Scalar expansion
- Row vs column vector broadcasting
- Incompatible shapes
- Explicit dimension control using `np.newaxis`

Outcome:
- Understand implicit dimension expansion
- Replace loops with vectorized operations
- Predict broadcast compatibility

Status: Completed

---

## 04 — Monte Carlo Coin Simulation

Focus:
- Random sampling with NumPy
- Vectorized simulation of repeated experiments
- Empirical probability estimation
- Law of Large Numbers demonstration
- Statistical visualization using Matplotlib

Implementation:
- Simulate 10,000 experiments
- Each experiment contains 100 coin flips
- Estimate head probability per trial
- Visualize probability distribution
- Demonstrate convergence to 0.5

Outcome:
- Apply shape, axis, and broadcasting knowledge in practice
- Understand Monte Carlo simulation basics
- Observe statistical convergence behavior
- Connect numerical computing with probability theory

Status: Completed

---

# Roadmap

Current Progress:

- [x] 01 Shape Experiments
- [x] 02 Axis Tests
- [x] 03 Broadcasting Cases
- [x] 04 Statistical Simulation

Potential future expansions:
- Biased coin simulation
- Binomial distribution comparison
- Central Limit Theorem experiment
- Performance comparison (vectorized vs loops)
- Portfolio-level statistical mini projects
  
---

# What This Project Demonstrates

This lab shows:

- Structured learning progression
- Clear conceptual separation of experiments
- Strong shape reasoning ability
- Understanding of vectorization
- Basic statistical simulation capability
- Clean, reproducible experimental design

It demonstrates foundational competence in numerical computing —
a prerequisite for machine learning frameworks such as:

- PyTorch
- TensorFlow
- JAX

---

# Why This Matters

Many learners can *use* NumPy.

Few understand:

- Why reshape fails
- Why ravel modifies original data
- Why axis collapses dimensions
- How broadcasting actually works

This repository aims to close that gap.

By the end of this lab, the reader should be able to:

- Predict tensor shapes mentally
- Reason about dimensional transformations
- Write vectorized numerical code confidently
- Prepare for advanced machine learning frameworks

---
## License
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
This project is licensed under the MIT License.  
See the [LICENSE](LICENSE) file for details.

# Usage

Clone the repository and run individual experiments:

```bash
git clone https://github.com/JoonKeat-MS/Project-NumPy-Structural-Foundations-Lab.git
cd Project-NumPy-Structural-Foundations-Lab
python 01_shape_experiments.py
python 02_axis_tests.py
python 03_broadcasting_cases.py
python 04_statistical_simulation.py




