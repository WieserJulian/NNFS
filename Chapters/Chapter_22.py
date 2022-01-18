import numpy as np
import matplotlib.pyplot as plt
import cv2

# Loads a MNIST dataset
from Classes.LoadData import create_data_mnist
from Classes.Models import Model

# # Create dataset
# X, y, X_test, y_test = create_data_mnist('fashion_mnist_images')
# # Scale and reshape samples
# X_test = (X_test.reshape(X_test.shape[0], -1).astype(np.float32) -
#           127.5) / 127.5
# # Load the model
# model = Model.load('fashion_mnist.model')
# # Predict on the first 5 samples from validation dataset
# # and print the result
# confidences = model.predict(X_test[:5])
# print(confidences)


# Label index to label name relation
fashion_mnist_labels = {
0: 'T-shirt/top',
1: 'Trouser',
2: 'Pullover',
3: 'Dress',
4: 'Coat',
5: 'Sandal',
6: 'Shirt',
7: 'Sneaker',
8: 'Bag',
9: 'Ankle boot'
}


def predictImage(image_data):
    # Resize to the same size as Fashion MNIST images
    image_data = cv2.resize(image_data, (28, 28))

    # Invert image colors
    image_data = 255 - image_data

    # Reshape and scale pixel data
    image_data = (image_data.reshape(1, -1).astype(np.float32) - 127.5) / 127.5
    # Load the model
    model = Model.load('fashion_mnist.model')
    # Predict on the image
    predictions = model.predict(image_data)
    # Get prediction instead of confidence levels
    predictions = model.output_layer_activation.predictions(predictions)
    # Get label name from label index
    prediction = fashion_mnist_labels[predictions[0]]
    print(prediction)

# Read an image
image_data = cv2.imread('TestData\Tshirt.png', cv2.IMREAD_GRAYSCALE)
predictImage(image_data)

# Read an image
image_data = cv2.imread('TestData\pants.png', cv2.IMREAD_GRAYSCALE)
predictImage(image_data)