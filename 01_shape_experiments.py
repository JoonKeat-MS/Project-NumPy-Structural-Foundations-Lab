"""
File: 01_shape_experiments.py
Author: Joon Keat Lim (JKai)
Date: 2026-02-26

Project: NumPy Structural Foundations Lab

Description:
This script explores fundamental structural properties of NumPy arrays,
including shape, dimensionality (ndim), reshape operations, axis reordering
via transpose, and memory behavior (view vs copy).

Objectives:
1. Understand how NumPy represents multidimensional arrays in memory.
2. Verify why reshape cannot change total element count.
3. Analyze axis permutation in 2D and 3D tensors.
4. Investigate memory sharing differences between flatten, ravel, and reshape.
5. Use np.shares_memory() to validate view vs copy behavior.

This file is part of a structured NumPy laboratory series designed to
build numerical computing intuition for statistical modeling and AI research.
"""

import numpy as np

print("========== SHAPE EXPERIMENTS ==========\n")
print("========== Arrays Creation ==========\n")
# ==========================================================
# Create arrays with different dimensions
# ==========================================================

a_1d = np.arange(6) # 1D Array from 0 to 5
a_2d = np.arange(6).reshape(2, 3) # 2D Array from 0 to 5
a_3d = np.arange(24).reshape(2, 3, 4) # 3D Array from 0 to 23

print("1D Array:\n", a_1d)
print("Shape:", a_1d.shape, " | Dimension:", a_1d.ndim)
print()

print("2D Array:\n", a_2d)
print("Shape:", a_2d.shape, " | Dimension:", a_2d.ndim)
print()

print("3D Array:\n", a_3d)
print("Shape:", a_3d.shape, " | Dimension:", a_3d.ndim)
print("\n--------------------------------------------------\n")

# ==========================================================
# Reshape experiments
# ==========================================================
print("========== Reshape Arrays ==========\n")

print("Reshape 1D -> 3x2")
b = a_1d.reshape(3, 2)
print(b)
print("Shape:", b.shape)
print("Dimension:", b.ndim)
print()

print("Reshape 1D -> 2x3")
c = a_1d.reshape(2, 3)
print(c)
print("Shape:", c.shape)
print("Dimension:", c.ndim)
print()

print("Reshape 1D -> 6x1x1")
d = a_1d.reshape(6, 1, 1)
print(d)
print("Shape:", d.shape)
print("Dimension:", d.ndim)
print()

print("Reshape 1D -> 1x1x6")
e = a_1d.reshape(1, 1, 6)
print(e)
print("Shape:", e.shape)
print("Dimension:", e.ndim)
print()

# Using -1 (automatic inference)
f = a_1d.reshape(-1, 1)
print("Reshape with -1:")
print(f)
print("Shape:", f.shape)
print("Dimension:", f.ndim)

print("\n--------------------------------------------------\n")

# ==========================================================
# Transpose (axis swapping)
# ==========================================================
print("========== Transpose Arrays ==========\n")

print("Original 2D:\n", a_2d)
print("Transposed 2D:\n", a_2d.T)
print("\nOriginal 2D shape:", a_2d.shape)
print("Transposed 2D shape:", a_2d.T.shape)
print()

print("Original 3D:\n", a_3d)
print("\nTransposed 3D (swap axis 0 and 1):\n", np.transpose(a_3d, (1, 0, 2)))
print("\nOriginal 3D shape:", a_3d.shape)
print("Transposed 3D shape (swap axis 0 and 1):", np.transpose(a_3d, (1, 0, 2)).shape)

print("\n--------------------------------------------------\n")

# ==========================================================
# Flatten vs Ravel vs Reshape (view vs copy)
# ==========================================================
print("========== Flatten vs Ravel vs Reshape ==========\n")

print("Original 2D:\n", a_2d)

flat = a_2d.flatten()
rav = a_2d.ravel()
reshape = a_2d.reshape(-1)

print("\nFlatten result:", flat)
print("Ravel result:", rav)
print("Reshape result:", reshape)

print("\nBase of flatten:", flat.base)
print("Base of ravel:", rav.base)
print("Base of reshape:", reshape.base)

print("\nModify ravel[0] = 999")
rav[0] = 999
print("Original after ravel modification:\n", a_2d)

print("\nModify flatten[1] = 555")
flat[1] = 555
print("Original after flatten modification:\n", a_2d)

print("\nModify Reshape[2] = 111")
reshape[2] = 111
print("Original after reshape modification:\n", a_2d)

print("\n--------------------------------------------------\n")

# ==========================================================
# Memory layout check
# ==========================================================
print("========== Memory Layout ==========\n")

print("Is ravel a view? (shares memory):",
      np.shares_memory(a_2d, rav))

print("Is flatten a view? (shares memory):",
      np.shares_memory(a_2d, flat))

print("Is reshape a view? (shares memory):",
      np.shares_memory(a_2d, reshape))

print("\n========== END OF SHAPE EXPERIMENT ==========")