# NumPy Structural Foundations Lab

## Overview
This repository is part of a structured NumPy laboratory series designed to build numerical computing intuition for statistical modeling and AI research.  
It explores how NumPy arrays behave in terms of **shape, dimensionality, axis manipulation, and memory sharing**.

## Objectives
- Understand how NumPy represents multidimensional arrays in memory.
- Verify why `reshape` cannot change the total element count.
- Analyze axis permutation in 2D and 3D tensors using `transpose`.
- Investigate memory sharing differences between `flatten`, `ravel`, and `reshape`.
- Use `np.shares_memory()` to validate view vs copy behavior.

## Contents
- `01_shape_experiments.py`  
  Demonstrates:
  - Array creation in 1D, 2D, and 3D
  - Reshape operations with explicit and inferred dimensions
  - Axis reordering with `transpose`
  - Comparison of `flatten`, `ravel`, and `reshape`
  - Memory layout checks with `np.shares_memory`
 
## Usage
Clone the repository and run the script:
```bash
git clone https://github.com/JoonKeat-MS/Project-NumPy-Structural-Foundations-Lab.git
cd Project-NumPy-Structural-Foundations-Lab
python 01_shape_experiments.py

## License
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This project is licensed under the MIT License.  
See the [LICENSE](LICENSE) file for details.

## Usage
Clone the repository and run the script:
```bash
git clone https://github.com/JoonKeat-MS/Project-NumPy-Structural-Foundations-Lab.git
cd Project-NumPy-Structural-Foundations-Lab
python 01_shape_experiments.py

---


