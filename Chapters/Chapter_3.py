import nnfs
import numpy as np
from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt

from Classes.Layers import Layer_Dense
from Classes.Loss import Loss
from Classes.Activation import Activation_Softmax, Activation_ReLU


inputs = [[1, 2, 3, 2.5],
          [2., 5., -1., 2],
          [-1.5, 2.7, 3.3, -0.8]]
weights = [[0.2, 0.8, -0.5, 1],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]
biases = [2, 3, 0.5]
weights2 = [[0.1, -0.14, 0.5],
            [-0.5, 0.12, -0.33],
            [-0.44, 0.73, -0.13]]
biases2 = [-1, 2, -0.5]

layer1_outputs = np.dot(inputs, np.array(weights).T) + biases
layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T) + biases2

print(layer2_outputs)
print("1")

nnfs.init()
X, y = spiral_data(samples=100, classes=3)
plt.scatter(X[:, 0], X[:, 1])
plt.show()
print("2")
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='brg')
plt.show()
print("3")
nnfs.init()
print(np.random.randn(2, 5))
print("4")



X, y = spiral_data(samples=100, classes=3)
dense1 = Layer_Dense(2, 3)
dense1.forward(X)
print(dense1.output[:5])
print("5")
