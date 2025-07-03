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
ALPHA = 0.001 / 10 /  5  #* 5 / 10
BATCH_SIZE = 8

KERNEL_SIZE = 3
STRIDE = 1

WEIGHTS_0_1_FILE_NAME = 'weights_0_1.npy'
WEIGHTS_1_2_FILE_NAME = 'weights_1_2.npy'
KERNEL_FILE_NAME = 'kernel.npy'

#Initialize the weights for layer 0_1 and load any existing weights for the layer if they exist
weights_0_1 = np.random.random((INPUT_SIZE, LAYER_1_SIZE)) - 0.5
if os.path.exists(WEIGHTS_0_1_FILE_NAME):
    weights_0_1 = np.load(WEIGHTS_0_1_FILE_NAME)

weights_1_2 = np.random.random((LAYER_1_SIZE, OUTPUT_SIZE)) - 0.5
if os.path.exists(WEIGHTS_1_2_FILE_NAME):
    weights_1_2 = np.load(WEIGHTS_1_2_FILE_NAME)

kernel = (np.random.random((KERNEL_SIZE, KERNEL_SIZE)) - 0.5) * 0.5
if os.path.exists(KERNEL_FILE_NAME):
    kernel = np.load(KERNEL_FILE_NAME)

num_batches_processed = 1
batch_count = 0
batch_gradient_1_2 = 0
batch_gradient_0_1 = 0
batch_gradient_kernel = np.zeros(kernel.shape)

num_correct = 0

for times in range (3):
    for iteration in range(6000):
        iteration += random.randint(0, 1)

        #Pad the image for convolution
        padded_img = np.zeros([IMG_LEN + 2, IMG_LEN + 2])
        padded_img[1:IMG_LEN + 1, 1:IMG_LEN + 1] = train_images[iteration] / 255.0

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

        # Find prediction index for debugging
        prediction_val = 0
        prediction_idx = 0
        for j in range(OUTPUT_SIZE):
            if probabilities[0][j] > prediction_val:
                prediction_val = probabilities[0][j]
                prediction_idx = j

        goal_pred = np.zeros((1, OUTPUT_SIZE))
        goal_pred[0][train_labels[iteration]] = 1.0

        # Calculate cross entropy error for output
        delta_output = probabilities - goal_pred
        error_output = - np.log(probabilities[0][train_labels[iteration]])

        batch_count = batch_count + 1



        # Update weights_1_2
        weighted_delta_output = np.dot(layer_1.T, delta_output)  # Scale each delta by the input that caused it. The transverse causes the resulting matrix shape to be the same as the weights
        batch_gradient_1_2 += weighted_delta_output

        # Chain rule the derivative between the hidden layer to output layer by multiplying by the delta of the input layer to hidden layer
        delta_layer_0_1 = np.dot(delta_output, weights_1_2.T) * activation_functions.tanh_derivative(layer_1)
        delta_layer_0_1 *= dropout_vector
        batch_gradient_0_1 += curr_input.T.dot(delta_layer_0_1)

        # Chain rule del layer 1 del kernel with the activation function of the convolution and previous derivatives for the fully connected layer
        delta_con_0 = np.dot(weights_0_1, delta_layer_0_1.T)
        delta_con_0 = delta_con_0.T
        delta_con_0 = delta_con_0.reshape(IMG_LEN, IMG_LEN) * activation_functions.tanh_derivative(convoluted_img)
        delta_kernel = np.zeros(kernel.shape)

        #Calculates the sum loss of the kernel for each pixel it produced
        for row in range(IMG_LEN):
            for col in range(IMG_LEN):

                #Get the original batch used at this point in the convolution
                region = padded_img[row:row + KERNEL_SIZE, col:col + KERNEL_SIZE]

                #Multiply the region matrix by the delta with respect to what the convolution produced for that pixel
                delta_kernel += region * delta_con_0[row, col]

        #Updates the kernel values
        #kernel = kernel - (ALPHA * delta_kernel)
        batch_gradient_kernel += delta_kernel

        if batch_count == BATCH_SIZE:
            weights_1_2 = weights_1_2 - (ALPHA * batch_gradient_1_2 / BATCH_SIZE)
            weights_0_1 = weights_0_1 - (ALPHA * batch_gradient_0_1 / BATCH_SIZE)
            kernel = kernel - (ALPHA * batch_gradient_kernel / BATCH_SIZE)

            num_batches_processed += 1

            batch_gradient_1_2 = 0
            batch_gradient_0_1 = 0
            batch_gradient_kernel = np.zeros(kernel.shape)
            batch_count = 0
            print(times)
            # print("Iteration:", iteration, "Prediction: " + str(prediction_idx) + " confidence: " + str(probabilities[0][prediction_idx] * 100) + "% correct answer: " + str(train_labels[iteration]) + get_result_str(prediction_idx, train_labels[iteration]))

        if prediction_idx == train_labels[iteration]:
            num_correct = num_correct + 1


        #print("Accuracy:", 100 * num_correct / (iteration + 1 + (times * 6000)), "Batch:", num_batches_processed, "Epoch Number:", times)



np.save(WEIGHTS_0_1_FILE_NAME, weights_0_1)
np.save(WEIGHTS_1_2_FILE_NAME, weights_1_2)
np.save(KERNEL_FILE_NAME, kernel)