import os
from random import random

import activation_functions
import file_utils

import math
import random

import numpy as np

def rand_neg():
    num = random.randint(0, 1)
    if num == 1:
        return -1
    else:
        return 1

def convert_img_to_input(dataset):
    input_layer = np.empty((1, INPUT_SIZE))

    for i in range(IMG_LEN):
        for j in range(IMG_LEN):
            input_layer[0][(i * IMG_LEN) + j] = dataset[i][j]

    return input_layer

# TODO: do numerically stable version
def softmax(last_layer, output_size):
    probabilities = np.zeros((1, OUTPUT_SIZE))
    denom = 0
    for j in range(output_size):
        val = math.exp(last_layer[0][j])
        denom += val
    for j in range(output_size):
        probabilities[0][j] = math.exp(last_layer[0][j]) / denom
    return probabilities

def get_result_str(pred_idx, ans_idx):
    if pred_idx == ans_idx:
        return #"\033[32mCorrect\033[0m"
    else:
        return "\033[31mIncorrect\033[0m"

IMG_LEN = 28
INPUT_SIZE = 784
LAYER_1_SIZE = 96
OUTPUT_SIZE = 10

KERNEL_SIZE = 3
STRIDE = 1

WEIGHTS_0_1_FILE_NAME = 'weights_0_1.npy'
WEIGHTS_1_2_FILE_NAME = 'weights_1_2.npy'
KERNEL_FILE_NAME = 'kernel.npy'

# File paths to the downloaded files
test_images_path = 'dataset/t10k-images.idx3-ubyte'
test_labels_path = 'dataset/t10k-labels.idx1-ubyte'

# Loads the dataset
test_images = file_utils.load_mnist_images(test_images_path)
test_labels = file_utils.load_mnist_labels(test_labels_path)

if not os.path.exists(WEIGHTS_0_1_FILE_NAME):
    print(WEIGHTS_0_1_FILE_NAME, "not found")
    exit()
weights_0_1 = np.load(WEIGHTS_0_1_FILE_NAME)

if not os.path.exists(WEIGHTS_1_2_FILE_NAME):
    print(WEIGHTS_1_2_FILE_NAME, "not found")
    exit()
weights_1_2 = np.load(WEIGHTS_1_2_FILE_NAME)

if not os.path.exists(KERNEL_FILE_NAME):
    print(KERNEL_FILE_NAME, "not found")
    exit()
kernel = np.load(KERNEL_FILE_NAME)

num_correct = 0

for iteration in range(6000):

    # Pad the image for convolution
    padded_img = np.zeros([IMG_LEN + 2, IMG_LEN + 2])
    padded_img[1:IMG_LEN + 1, 1:IMG_LEN + 1] = test_images[iteration] / 255.0

    convoluted_img = np.zeros([IMG_LEN, IMG_LEN])

    for row in range(IMG_LEN):
        for col in range(IMG_LEN):
            sub_section = padded_img[row:row + KERNEL_SIZE, col:col + KERNEL_SIZE]

            convoluted_img[row][col] = np.sum(sub_section * kernel)

    # Applying non-linearity currently
    convoluted_img = activation_functions.tanh(convoluted_img)

    # Forward propagation
    curr_input = convert_img_to_input(convoluted_img)
    layer_1 = activation_functions.tanh(np.dot(curr_input, weights_0_1))
    dropout_vector = np.random.choice([0, 1], size=(1, LAYER_1_SIZE), p=[0.25, 0.75])
    layer_1 *= dropout_vector / 0.75
    output = np.dot(layer_1, weights_1_2)

    probabilities = softmax(output, OUTPUT_SIZE)
    # probabilities = output

    # Find prediction index for debugging
    prediction_val = 0
    prediction_idx = 0
    for j in range(OUTPUT_SIZE):
        if probabilities[0][j] > prediction_val:
            prediction_val = probabilities[0][j]
            prediction_idx = j

    goal_pred = np.zeros((1, OUTPUT_SIZE))
    goal_pred[0][test_labels[iteration]] = 1.0

    '''
    # Calculate mean squared error for output
    delta_output = output - goal_pred
    error_output = delta_output ** 2
    '''
    # Calculate cross entropy error for output
    delta_output = probabilities - goal_pred
    error_output = - np.log(probabilities[0][test_labels[iteration]])

    if prediction_idx == test_labels[iteration]:
        num_correct += 1

    #print("Iteration:", iteration, "Prediction: " + str(prediction_idx) + " confidence: " + str(probabilities[0][prediction_idx] * 100) + "% correct answer: " + str(test_labels[iteration]) + get_result_str(prediction_idx, test_labels[iteration]))

print("----------")
print('Accuracy: %', num_correct / 6000 * 100)