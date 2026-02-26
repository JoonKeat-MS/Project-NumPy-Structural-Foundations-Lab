"""
File: 02_axis_tests.py
Author: Joon Keat Lim (JKai)
Date: 2026-02-26

Project: NumPy Structural Foundations Lab

Description:
This script investigates the behavior of axis-based operations in NumPy.
It explores how reduction functions such as sum(), mean(), max(), and min()
operate across different axes and how dimensionality is affected.

Objectives:
1. Develop a precise understanding of the axis parameter in NumPy.
2. Observe how reduction operations collapse specific dimensions.
3. Compare row-wise vs column-wise aggregation behavior.
4. Analyze shape transformations after axis reductions.
5. Explore the effect of keepdims=True in preserving dimensions.

This file strengthens structural reasoning about multidimensional arrays,
which is essential for statistical computation, matrix operations,
and machine learning workflows.
"""

import numpy as np

print("========== AXIS TESTS ==========\n")

# =======================================================
# Create a 2D array for experiments
# =======================================================

A = np.array([
    [1, 2, 3, 4],
    [10, 20, 30, 40],
    [100, 200, 300, 400]
])

print("Original Array A:\n", A)
print("Shape:", A.shape)
print("Dimensions:", A.ndim)
print("\n--------------------------------------------------\n")

# =======================================================
# Sum along different axes
# =======================================================

print("========== Sum Operations ==========\n")

sum_axis0 = np.sum(A, axis=0)
print("Sum (axis=0) → Column-wise sum:")
print(sum_axis0)
print("Result Shape:", sum_axis0.shape)
print()

sum_axis1 = np.sum(A, axis=1)
print("Sum (axis=1) → Row-wise sum:")
print(sum_axis1)
print("Result Shape:", sum_axis1.shape)
print()

sum_all = np.sum(A)
print("Sum (no axis) → All elements:")
print(sum_all)
print("Result Type:", type(sum_all))
print("\n--------------------------------------------------\n")

# =======================================================
# Mean along different axes
# =======================================================

print("========== Mean Operations ==========\n")

print("Mean (axis=0):", np.mean(A, axis=0))
print("Mean (axis=1):", np.mean(A, axis=1))
print("\n--------------------------------------------------\n")

# =======================================================
# Max and Min
# =======================================================

print("========== Max/Min Operations ==========\n")

print("Max (axis=0):", np.max(A, axis=0))
print("Max (axis=1):", np.max(A, axis=1))
print()

print("Min (axis=0):", np.min(A, axis=0))
print("Min (axis=1):", np.min(A, axis=1))
print("\n--------------------------------------------------\n")

# =======================================================
# keepdims Behavior
# =======================================================

print("========== Keep Dimensions ==========\n")

keep0 = np.sum(A, axis=0, keepdims=True)
keep1 = np.sum(A, axis=1, keepdims=True)

print("Sum axis=0 with keepdims=True:")
print(keep0)
print("Shape:", keep0.shape)
print()

print("Sum axis=1 with keepdims=True:")
print(keep1)
print("Shape:", keep1.shape)
print("\n--------------------------------------------------\n")

# =======================================================
# 3D Axis Experiment
# =======================================================

print("========== 3D Axis Experiment ==========\n")

B = np.arange(24).reshape(2, 3, 4)

print("3D Array B:\n", B)
print("Shape:", B.shape)
print()

print("Sum axis=0 → collapse first dimension:")
print(np.sum(B, axis=0))
print("Shape:", np.sum(B, axis=0).shape)
print()

print("Sum axis=1 → collapse second dimension:")
print(np.sum(B, axis=1))
print("Shape:", np.sum(B, axis=1).shape)
print()

print("Sum axis=2 → collapse third dimension:")
print(np.sum(B, axis=2))
print("Shape:", np.sum(B, axis=2).shape)

print("\n========== END OF AXIS TESTS ==========")