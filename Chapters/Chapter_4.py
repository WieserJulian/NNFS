import math

import nnfs
import numpy as np
from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt

from Classes.Layers import Layer_Dense
from Classes.Loss import Loss
from Classes.Activation import Activation_Softmax, Activation_ReLU

print("1")
inputs = [0, 2, -1, 3.3, -2.7, 1.1, 2.2, -100]
output = []
for i in inputs:
    if i > 0:
        output.append(i)
    else:
        output.append(0)
print(output)

print("2")
inputs = [0, 2, -1, 3.3, -2.7, 1.1, 2.2, -100]
output = np.maximum(0, inputs)
print(output)

print("3")

# Create dataset
X, y = spiral_data(samples=100, classes=3)
# Create Dense layer with 2 input features and 3 output values
dense1 = Layer_Dense(2, 3)
# Create ReLU activation (to be used with Dense layer):
activation1 = Activation_ReLU()
# Make a forward pass of our training data through this layer
dense1.forward(X)
# Forward pass through activation func.
# Takes in output from previous layer
activation1.forward(dense1.output)
# Let's see output of the first few samples:
print(activation1.output[:5])

print("4")
# Values from the previous output when we described
# what a neural network is
layer_outputs = [4.8, 1.21, 2.385]
# e - mathematical constant, we use E here to match a common coding
# style where constants are uppercased
E = math.e
# For each value in a vector, calculate the exponential value
exp_values = []
for output in layer_outputs:
    exp_values.append(E ** output)  # ** - power operator in Python
print('exponentiated values:')
print(exp_values)

print("5")
# Now normalize values
norm_base = sum(exp_values)  # We sum all values
norm_values = []
for value in exp_values:
    norm_values.append(value / norm_base)
print('Normalized exponentiated values:')
print(norm_values)
print('Sum of normalized values:', sum(norm_values))

print("6")
# Values from the earlier previous when we described
# what a neural network is
layer_outputs = [4.8, 1.21, 2.385]
# For each value in a vector, calculate the exponential value
exp_values = np.exp(layer_outputs)
print('exponentiated values:')
print(exp_values)
# Now normalize values
norm_values = exp_values / np.sum(exp_values)
print('normalized exponentiated values:')
print(norm_values)
print('sum of normalized values:', np.sum(norm_values))

print("7")
layer_outputs = np.array([[4.8, 1.21, 2.385],
                          [8.9, -1.81, 0.2],
                          [1.41, 1.051, 0.026]])
print('Sum without axis')
print(np.sum(layer_outputs))
print('This will be identical to the above since default is None:')
print(np.sum(layer_outputs, axis=None))
print('Another way to think of it w/ a matrix == axis 0: columns:')
print(np.sum(layer_outputs, axis=0))

print("8")
print('But we want to sum the rows instead, like this w/ raw py:')
for i in layer_outputs:
    print(sum(i))

print("9")
print('So we can sum axis 1, but note the current shape:')
print(np.sum(layer_outputs, axis=1))

print("10")
print('Sum axis 1, but keep the same dimensions as input:')
print(np.sum(layer_outputs, axis=1, keepdims=True))

print("11")
# Create dataset
X, y = spiral_data(samples=100, classes=3)
# Create Dense layer with 2 input features and 3 output values
dense1 = Layer_Dense(2, 3)
# Create ReLU activation (to be used with Dense layer):
activation1 = Activation_ReLU()
# Create second Dense layer with 3 input features (as we take output
# of previous layer here) and 3 output values
dense2 = Layer_Dense(3, 3)
# Create Softmax activation (to be used with Dense layer):
activation2 = Activation_Softmax()
# Make a forward pass of our training data through this layer
dense1.forward(X)
# Make a forward pass through activation function
# it takes the output of first dense layer here
activation1.forward(dense1.output)
# Make a forward pass through second Dense layer
# it takes outputs of activation function of first layer as inputs
dense2.forward(activation1.output)
# Make a forward pass through activation function
# it takes the output of second dense layer here
activation2.forward(dense2.output)
# Let's see output of the first few samples:
print(activation2.output[:5])


