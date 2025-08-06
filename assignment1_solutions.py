
# Assignment-1 Solutions Based on NumPy

import numpy as np
from PIL import Image
from itertools import product

# Q1: Basic NumPy Array
print("Q1 (a): Reverse Array")
arr = np.array([1, 2, 3, 6, 4, 5])
print(arr[::-1])

print("\nQ1 (b): Flatten array using two methods")
array1 = np.array([[1, 2, 3], [2, 4, 5], [1, 2, 3]])
print(array1.flatten())
print(array1.ravel())

print("\nQ1 (c): Compare arrays")
arr1 = np.array([[1, 2], [3, 4]])
arr2 = np.array([[1, 2], [3, 4]])
print(np.array_equal(arr1, arr2))

print("\nQ1 (d): Most frequent value and indices")
x = np.array([1,2,3,4,5,1,2,1,1,1])
y = np.array([1,1,1,2,3,4,2,4,3,3])
for a in [x, y]:
    vals, counts = np.unique(a, return_counts=True)
    max_val = vals[np.argmax(counts)]
    indices = np.where(a == max_val)[0]
    print(f"Most frequent: {max_val}, Indices: {indices}")

print("\nQ1 (e): Matrix sums")
gfg = np.matrix('[4, 1, 9; 12, 3, 1; 4, 5, 6]')
print(np.sum(gfg))
print(np.sum(gfg, axis=1))
print(np.sum(gfg, axis=0))

print("\nQ1 (f): Matrix operations")
n_array = np.array([[55, 25, 15],[30, 44, 2],[11, 45, 77]])
print("Diagonal sum:", np.trace(n_array))
eig_vals, eig_vecs = np.linalg.eig(n_array)
print("Eigenvalues:", eig_vals)
print("Eigenvectors:\n", eig_vecs)
print("Inverse:\n", np.linalg.inv(n_array))
print("Determinant:", np.linalg.det(n_array))

print("\nQ1 (g): Matrix multiplication & covariance")
p1 = np.array([[1,2],[2,3]])
q1 = np.array([[4,5],[6,7]])
print("Product 1:\n", np.dot(p1, q1))
print("Covariance:\n", np.cov(p1, q1, rowvar=False))

p2 = np.array([[1,2],[2,3],[4,5]])
q2 = np.array([[4,5,1],[6,7,2]])
print("Product 2:\n", np.dot(p2, q2))

print("\nQ1 (h): Products")
x = np.array([[2, 3, 4], [3, 2, 9]])
y = np.array([[1, 5, 0], [5, 10, 3]])
print("Inner:\n", np.inner(x, y))
print("Outer:\n", np.outer(x.flatten(), y.flatten()))
print("Cartesian product:\n", list(product(x.flatten(), y.flatten())))

# Q2: Math and Stats
print("\nQ2 (a): Absolute and Percentiles")
array = np.array([[1, -2, 3],[-4, 5, -6]])
print("Absolute:\n", np.abs(array))
flat = array.flatten()
print("Percentiles (flat):", np.percentile(flat, [25, 50, 75]))
print("Percentiles (columns):", np.percentile(array, [25, 50, 75], axis=0))
print("Percentiles (rows):", np.percentile(array, [25, 50, 75], axis=1))

print("Mean:", np.mean(flat), "Median:", np.median(flat), "Std Dev:", np.std(flat))
print("Mean (rows):", np.mean(array, axis=1))
print("Mean (cols):", np.mean(array, axis=0))

print("\nQ2 (b): Floor, ceil, trunc, round")
a = np.array([-1.8, -1.6, -0.5, 0.5,1.6, 1.8, 3.0])
print("Floor:", np.floor(a))
print("Ceil:", np.ceil(a))
print("Trunc:", np.trunc(a))
print("Rounded:", np.round(a))

# Q3: Searching & Sorting
print("\nQ3 (a): Sorting")
array = np.array([10, 52, 62, 16, 16, 54, 453])
print("Sorted:", np.sort(array))
print("Indices:", np.argsort(array))
print("4 Smallest:", np.sort(array)[:4])
print("5 Largest:", np.sort(array)[-5:][::-1])

print("\nQ3 (b): Integer and Float elements")
array = np.array([1.0, 1.2, 2.2, 2.0, 3.0, 2.0])
print("Integer elements:", array[array == array.astype(int)])
print("Float elements:", array[array != array.astype(int)])

# Q4: Image handling
print("\nQ4 (a): img_to_array")

def img_to_array(path):
    img = Image.open(path)
    arr = np.array(img)
    if len(arr.shape) == 2:
        np.savetxt("grayscale_image.txt", arr, fmt="%d")
    else:
        np.savetxt("rgb_image.txt", arr.reshape(-1, arr.shape[2]), fmt="%d")
    return arr

print("Q4 (b): Load back")
def load_image_array(file_path):
    return np.loadtxt(file_path)
