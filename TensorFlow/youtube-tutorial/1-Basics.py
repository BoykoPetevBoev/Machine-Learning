import tensorflow as tf
import pandas as pd

print(pd.__version__)
print(tf.__version__)

x = tf.constant(4, shape=(1, 1))
x = tf.constant([[1, 2, 3], [4, 5, 6]], shape=(2, 3))
x = tf.eye(4)  # I for the identity matrix
x = tf.random.uniform((3, 3), dtype=float)
x = tf.range(start=1, limit=10, delta=2)

# Mathematical Operations
print("- Mathematical Operations")
x = tf.constant([1, 2, 3])
y = tf.constant([9, 8, 7])
z = tf.add(x, y)
print(z)
z = tf.subtract(x, y)
print(z)
z = tf.divide(x, y)
print(z)
z = tf.multiply(x, y)
print(z)

# Indexing
print("- Indexing")
x = tf.constant([0, 1, 1, 2, 3, 1, 2, 3])
print(x[:])
print(x[1:])
print(x[1:3])

# Reshaping
print("- Reshaping")
x = tf.range(9)
print(x)
x = tf.reshape(x, (3, 3))
print(x)
x = tf.transpose(x, perm=[1, 0])
print(x)
