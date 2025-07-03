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

def convert_img_to_input(idx, dataset):
    input_layer = np.empty((1, INPUT_SIZE))

    for i in range(IMG_LEN):
        for j in range(IMG_LEN):
            input_layer[0][(i * IMG_LEN) + j] = dataset[idx][i][j]

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
        return "\033[32mCorrect\033[0m"
    else:
        return "\033[31mIncorrect\033[0m"

# File paths to the downloaded files
train_images_path = 'dataset/train-images.idx3-ubyte'
train_labels_path = 'dataset/train-labels.idx1-ubyte'

# Loads the dataset
train_images = file_utils.load_mnist_images(train_images_path)
train_labels = file_utils.load_mnist_labels(train_labels_path)

# Check the shapes
print("Train images shape:", train_images.shape)  # Should print (60000, 28, 28)
print("Train labels shape:", train_labels.shape)  # Should print (60000,)

IMG_LEN = 28
INPUT_SIZE = 784
LAYER_1_SIZE = 96
OUTPUT_SIZE = 10
ALPHA = 0.001
BATCH_SIZE = 5

WEIGHTS_0_1_FILE_NAME = 'weights_0_1.npy'
WEIGHTS_1_2_FILE_NAME = 'weights_1_2.npy'

weights_0_1 = np.random.random((INPUT_SIZE, LAYER_1_SIZE)) - 0.5
if os.path.exists(WEIGHTS_0_1_FILE_NAME):
    weights_0_1 = np.load(WEIGHTS_0_1_FILE_NAME)

weights_1_2 = np.random.random((LAYER_1_SIZE, OUTPUT_SIZE)) - 0.5
if os.path.exists(WEIGHTS_1_2_FILE_NAME):
    weights_1_2 = np.load(WEIGHTS_1_2_FILE_NAME)

batch_count = 0
batch_delta_output = np.zeros((1,10))

num_correct = 0

for times in range (15):
    for iteration in range(6000):

        # Forward propagation
        curr_input = convert_img_to_input(iteration, train_images) / 255.0
        layer_1 = activation_functions.tanh(np.dot(curr_input, weights_0_1))
        #dropout_vector = np.random.randint(0, 2, (1,LAYER_1_SIZE))
        dropout_vector = np.random.choice([0, 1], size=(1, LAYER_1_SIZE), p=[0.5, 0.5])
        layer_1 * dropout_vector / 0.5
        output = np.dot(layer_1, weights_1_2)

        probabilities = softmax(output, OUTPUT_SIZE)
        #probabilities = output

        # Find prediction index for debugging
        prediction_val = 0
        prediction_idx = 0
        for j in range(OUTPUT_SIZE):
            if probabilities[0][j] > prediction_val:
                prediction_val = probabilities[0][j]
                prediction_idx = j

        goal_pred = np.zeros((1, OUTPUT_SIZE))
        goal_pred[0][train_labels[iteration]] = 1.0

        '''
        # Calculate mean squared error for output
        delta_output = output - goal_pred
        error_output = delta_output ** 2
        '''
        # Calculate cross entropy error for output
        delta_output = probabilities - goal_pred
        error_output = - np.log(probabilities[0][train_labels[iteration]])

        batch_delta_output = batch_delta_output + delta_output
        batch_count = batch_count + 1

        if batch_count == BATCH_SIZE:

            avg_batch_delta_output = batch_delta_output / BATCH_SIZE

            # Update weights_1_2
            weighted_delta_output = layer_1.T.dot(avg_batch_delta_output)  # Scale each delta by the input that caused it. The transverse causes the resulting matrix shape to be the same as the weights
            weights_1_2 = weights_1_2 - (ALPHA * weighted_delta_output)

            # Chain rule the derivative between the hidden layer to output layer by multiplying by the delta of the input layer to hidden layer
            delta_layer_0_1 = np.dot(avg_batch_delta_output, weights_1_2.T) * activation_functions.tanh_derivative(layer_1)
            weights_0_1 = weights_0_1 - (ALPHA * curr_input.T.dot(delta_layer_0_1))

            # print("Iteration:", iteration, "Prediction: " + str(prediction_idx) + " confidence: " + str(probabilities[0][prediction_idx] * 100) + "% correct answer: " + str(train_labels[iteration]) + get_result_str(prediction_idx, train_labels[iteration]))

            if prediction_idx == train_labels[iteration]:
                num_correct = num_correct + 1

            print("Accuracy:", 100 * num_correct / ((iteration / BATCH_SIZE) + 1 + (times * 6000 / BATCH_SIZE)))

            batch_delta_output = batch_delta_output * 0
            batch_count = 0


np.save(WEIGHTS_0_1_FILE_NAME, weights_0_1)
np.save(WEIGHTS_1_2_FILE_NAME, weights_1_2)