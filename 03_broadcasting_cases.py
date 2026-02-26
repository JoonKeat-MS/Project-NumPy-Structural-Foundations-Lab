"""
File: 03_broadcasting_cases.py
Author: Joon Keat Lim (JKai)
Date: 2026-02-26

Project: NumPy Structural Foundations Lab

Description:
This script investigates NumPy broadcasting mechanics.
It explores how arrays of different shapes interact during
element-wise operations without explicit replication.

Objectives:
1. Understand NumPy broadcasting rules.
2. Predict shape compatibility before computation.
3. Analyze implicit dimension expansion.
4. Compare scalar, vector, and matrix broadcasting.
5. Strengthen mental shape-tracing ability.

This experiment builds structural intuition for
vectorized numerical computation and AI workloads.
"""

import numpy as np

print("========== BROADCASTING CASES ==========\n")

# =======================================================
# Scalar Broadcasting
# =======================================================

print("========== Scalar Broadcasting ==========\n")

A = np.array([[1, 2, 3],
              [4, 5, 6]])

print("Matrix A:\n", A)
print("Shape:", A.shape)

result_scalar = A + 10
print("\nA + 10:\n", result_scalar)
print("Shape:", result_scalar.shape)

print("\n--------------------------------------------------\n")

# =======================================================
# Vector to Matrix Broadcasting (Row-wise)
# =======================================================

print("========== Row Vector Broadcasting ==========\n")

row_vec = np.array([10, 20, 30])

print("Row Vector:", row_vec)
print("Shape:", row_vec.shape)

result_row = A + row_vec
print("\nA + row_vec:\n", result_row)
print("Result Shape:", result_row.shape)

print("\n--------------------------------------------------\n")

# =======================================================
# Column Vector Broadcasting
# =======================================================

print("========== Column Vector Broadcasting ==========\n")

col_vec = np.array([[10],
                    [20]])

print("Column Vector:\n", col_vec)
print("Shape:", col_vec.shape)

result_col = A + col_vec
print("\nA + col_vec:\n", result_col)
print("Result Shape:", result_col.shape)

print("\n--------------------------------------------------\n")

# =======================================================
# Incompatible Shapes
# =======================================================

print("========== Incompatible Shapes ==========\n")

B = np.array([1, 2])

print("Vector B:", B)
print("Shape:", B.shape)

try:
    print(A + B)
except ValueError as e:
    print("Broadcasting Error:", e)

print("\n--------------------------------------------------\n")

# =======================================================
# 3D Broadcasting
# =======================================================

print("========== 3D Broadcasting ==========\n")

X = np.arange(24).reshape(2, 3, 4)
Y = np.array([1, 2, 3, 4])

print("3D Tensor X shape:", X.shape)
print("Vector Y shape:", Y.shape)

result_3d = X + Y
print("\nResult Arrays:", result_3d)
print("\nResult Shape:", result_3d.shape)

print("\n--------------------------------------------------\n")

# =======================================================
# Using np.newaxis for Explicit Expansion
# =======================================================

print("========== Explicit Dimension Expansion ==========\n")

vec = np.array([1, 2, 3])
print("Original Vector Shape:", vec.shape)

expanded_axis1 = vec[:, np.newaxis]
print("\nExpanded_Axis1 Vector:\n", expanded_axis1)
print("Expanded_Axis1 Shape:", expanded_axis1.shape)

expanded_axis0 = vec[np.newaxis, :]
print("\nExpanded_Axis2 Vector:\n", expanded_axis0)
print("Expanded_Axis2 Shape:", expanded_axis0.shape)

# Exlicit BroadCasting Example
a_1 = np.array([1, 2, 3])
a_2 = np.array([10, 100])

try:
    print(a_1 + a_2)
except ValueError as e1:
    print("\nBroadcasting Error:", e1)
    print("This will cause the operation of (1,3) and (1,2) which causes errors")

print("\nTo solve the issue, we could try to use explicit broadcast as below:")
print("a) Make a_2 shape from (2,) to (2, 1)")
expanded_a_2 = a_2[:, np.newaxis]
print(a_1 + expanded_a_2)

print("\nb) Make a_1 shape from (3,) to (3,1)")
expanded_a_1 = a_1[:,np.newaxis]
print(expanded_a_1 + a_2)

print("/nThe approaches are useful and important in many fields such as ML, DL and AI")

print("\n========== END OF BROADCASTING TESTS ==========")