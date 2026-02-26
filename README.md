# NumPy Structural Foundations Lab

## Overview

This repository is a structured experimental lab focused on building
deep structural intuition about NumPy arrays.

Rather than learning NumPy as an API collection, this project investigates
how multidimensional arrays behave internally — in terms of shape,
axis mechanics, broadcasting rules, and memory layout.

The ultimate goal is to develop strong numerical reasoning skills
for statistical modeling, scientific computing, and AI research.

---

# Core Learning Philosophy

NumPy is not just a library — it is a computational model.

Understanding it requires clarity in:

- Dimensional structure
- Axis semantics
- Memory representation
- Reduction logic
- Broadcasting rules

Each experiment in this repository isolates one structural concept
and studies it rigorously.

---

# Experimental Series

This lab consists of four structured experiments:

---

## 01 — Shape & Memory Foundations

Focus:
- Array creation (1D, 2D, 3D)
- Shape and dimensionality (`shape`, `ndim`)
- `reshape` mechanics and element invariance
- Axis permutation using `transpose`
- Memory behavior: `flatten` vs `ravel` vs `reshape`
- Verifying view vs copy using `np.shares_memory`

Outcome:
- Understand why total element count cannot change
- Understand how tensors are reinterpreted structurally
- Understand memory-sharing behavior

Status: Completed

---

## 02 — Axis & Reduction Semantics

Focus:
- Meaning of `axis`
- Row-wise vs column-wise aggregation
- Reduction operations (`sum`, `mean`, `max`, `min`)
- Dimensional collapse behavior
- `keepdims=True` and shape preservation
- 3D axis reduction experiments

Outcome:
- Predict result shapes without running code
- Understand how statistical reductions operate on tensors
- Develop shape reasoning across dimensions

Status: Completed

---

## 03 — Broadcasting & Vectorization (Upcoming)

Focus:
- Broadcasting rules
- Implicit dimension expansion
- Shape alignment algorithm
- Scalar vs vector vs matrix operations
- Practical vectorized computation

Outcome:
- Replace loops with vectorized logic
- Predict broadcast compatibility
- Understand dimension auto-expansion

Status: Completed

---

## 04 — Statistical Simulation & Numerical Modeling (Upcoming)

Focus:
- Random sampling (`np.random`)
- Monte Carlo simulation
- Empirical probability estimation
- Vectorized statistical computation
- Performance comparison vs pure Python loops

Outcome:
- Use NumPy for real statistical modeling
- Implement efficient simulation pipelines
- Apply structural knowledge to computation

Status: In Progress

---

# Roadmap

Current Progress:

- [x] 01 Shape Experiments
- [x] 02 Axis Tests
- [x] 03 Broadcasting Cases
- [ ] 04 Statistical Simulation

Next Milestone:
Complete broadcasting mastery and apply it in a Monte Carlo simulation project.

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





