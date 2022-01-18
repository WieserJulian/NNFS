import numpy as np
import cv2
import os

# Loads a MNIST dataset
from Classes.Accuracy import Accuracy_Categorical
from Classes.Activation import Activation_ReLU, Activation_Softmax
from Classes.Layers import Layer_Dense
from Classes.LoadData import create_data_mnist
from Classes.Loss import Loss_CategoricalCrossentropy
from Classes.Models import Model
from Classes.Optimizer import Optimizer_Adam

# Create dataset
X, y, X_test, y_test = create_data_mnist('fashion_mnist_images')
# Shuffle the training dataset
keys = np.array(range(X.shape[0]))
np.random.shuffle(keys)
X = X[keys]
y = y[keys]
# Scale and reshape samples
X = (X.reshape(X.shape[0], -1).astype(np.float32) - 127.5) / 127.5
X_test = (X_test.reshape(X_test.shape[0], -1).astype(np.float32) -
          127.5) / 127.5
# Instantiate the model
model = Model()
# Add layers
model.add(Layer_Dense(X.shape[1], 128))
model.add(Activation_ReLU())
model.add(Layer_Dense(128, 128))
model.add(Activation_ReLU())
model.add(Layer_Dense(128, 10))
model.add(Activation_Softmax())
# Set loss, optimizer and accuracy objects
model.set(
    loss=Loss_CategoricalCrossentropy(),
    optimizer=Optimizer_Adam(decay=1e-3),
    accuracy=Accuracy_Categorical()
)
# Finalize the model
model.finalize()
# Train the model
model.train(X, y, validation_data=(X_test, y_test),
            epochs=10, batch_size=128, print_every=100)
model.evaluate(X_test, y_test)
model.evaluate(X, y)
