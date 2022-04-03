import numpy as np

inputs = [1, 2, 3, 4.5]
weights = [0.2, 0.5, -0.5, 1.0]
bias = 2

output = np.dot(inputs, weights) + bias
print(output)