import os

import activation_functions
import file_utils
import utils

import numpy as np

# The side length of the square image
IMG_LEN = 28

# The side length of the kernel and stride for the convolutional layer
KERNEL_LEN = 3
STRIDE = 1

# The shape of the fully connected layers of the network
INPUT_SIZE = 784
LAYER_1_SIZE = 96
OUTPUT_SIZE = 10

WEIGHTS_0_1_FILE_NAME = 'weights_0_1.npy'
WEIGHTS_1_2_FILE_NAME = 'weights_1_2.npy'
KERNEL_FILE_NAME = 'kernel.npy'

# File paths to the downloaded files
test_images_path = 'dataset/t10k-images.idx3-ubyte'
test_labels_path = 'dataset/t10k-labels.idx1-ubyte'

# Loads the testing data
test_images = file_utils.load_mnist_images(test_images_path)
test_labels = file_utils.load_mnist_labels(test_labels_path)

# Initialize the weights for the kernel and load an existing kernel if it exists
if not os.path.exists(KERNEL_FILE_NAME):
    print(KERNEL_FILE_NAME, "not found")
    exit()
kernel = np.load(KERNEL_FILE_NAME)

# Initialize the weights for layer 0_1 and load any existing weights for the layer if they exist
if not os.path.exists(WEIGHTS_0_1_FILE_NAME):
    print(WEIGHTS_0_1_FILE_NAME, "not found")
    exit()
weights_0_1 = np.load(WEIGHTS_0_1_FILE_NAME)

# Initialize the weights for layer 1_2 and load any existing weights for the layer if they exist
if not os.path.exists(WEIGHTS_1_2_FILE_NAME):
    print(WEIGHTS_1_2_FILE_NAME, "not found")
    exit()
weights_1_2 = np.load(WEIGHTS_1_2_FILE_NAME)

# Stores the number of correct predictions
num_correct = 0

# Tests the model against the testing data
for iteration in range(test_labels.size):
    '''
    FORWARD PASS
    '''

    # Pad the image for convolution
    padded_img = np.zeros([IMG_LEN + 2, IMG_LEN + 2])
    padded_img[1:IMG_LEN + 1, 1:IMG_LEN + 1] = test_images[iteration] / 255.0

    # Stores the convolution image
    convoluted_img = np.zeros([IMG_LEN, IMG_LEN])

    # Calculates the convolution
    for row in range(IMG_LEN):
        for col in range(IMG_LEN):
            # Get a kernel sized subsection of the image
            subsection = padded_img[row:row + KERNEL_LEN, col:col + KERNEL_LEN]

            # Take the sum of the subsection and store it in the convoluted image
            convoluted_img[row][col] = np.sum(subsection * kernel)

    # Apply non-linearity after calculating the convolution and then flatten
    convoluted_img = activation_functions.tanh(convoluted_img)
    layer_0 = utils.flatten_img(convoluted_img, INPUT_SIZE, IMG_LEN)

    # Forward propagate to layer 1
    layer_1 = activation_functions.tanh(np.dot(layer_0, weights_0_1))

    # Forward propagate to the output layer and calculate the probability distribution
    output = np.dot(layer_1, weights_1_2)
    probabilities = utils.softmax(output, OUTPUT_SIZE)

    # Calculates prediction index for monitoring
    prediction_val = 0
    prediction_idx = 0
    for j in range(OUTPUT_SIZE):
        if probabilities[0][j] > prediction_val:
            prediction_val = probabilities[0][j]
            prediction_idx = j

    # Converts the prediction index into a vector
    goal_pred = np.zeros((1, OUTPUT_SIZE))
    goal_pred[0][test_labels[iteration]] = 1.0

    '''
    CHECK RESULT
    '''

    # Calculate cross entropy error for output
    delta_output = probabilities - goal_pred
    error_output = - np.log(probabilities[0][test_labels[iteration]])

    # Increment the number of correct predictions if the model predicted correctly
    if prediction_idx == test_labels[iteration]:
        num_correct += 1

    # Prints the result of this pass
    print("Iteration:", iteration, "Prediction: " + str(prediction_idx) + " confidence: " + str(probabilities[0][prediction_idx] * 100) + "% correct answer: " + str(test_labels[iteration]) + utils.get_result_str(prediction_idx, test_labels[iteration]))

# Prints the final accuracy of the model
print("----------")
print('Accuracy: %', 100 * num_correct / test_labels.size)