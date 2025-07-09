import math
import random

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

def rand_neg():
    """
    :return: A random integer; Either 1 or -1
    """
    num = random.randint(0, 1)
    if num == 1:
        return -1
    else:
        return 1

def flatten_img(img, flatten_size, img_len):
    """
    Manually flattens a 2-dimensional square image array
    :param img: 2-dimensional image array
    :param flatten_size: The size of the flattened array
    :param img_len: The side length the image
    :return: a 2-dimensional square image array
    """

    # Stores the new flattened layer
    flattened_layer = np.empty((1, flatten_size))

    # Moves each value to the correct spot in the flattened array through itteration
    for i in range(img_len):
        for j in range(img_len):
            flattened_layer[0][(i * img_len) + j] = img[i][j]

    # Returns the flattened layer
    return flattened_layer

# TODO: do numerically stable version
def softmax(last_layer, layer_size):
    """
    Converts the result of the network into a probability distribution
    :param last_layer: The final layer of the network
    :param layer_size: The size of the final layer
    :return: A probability distribution of the last layer
    """

    # Stores the probability distribution
    probabilities = np.zeros((1, layer_size))

    # Sum e raised to the exponent of each output to calculate the denominator
    denom = 0
    for j in range(layer_size):
        val = math.exp(last_layer[0][j])
        denom += val

    # Calculate each probability
    for j in range(layer_size):
        probabilities[0][j] = math.exp(last_layer[0][j]) / denom

    # Returns the probability distribution
    return probabilities

def get_result_str(pred_idx, ans_idx):
    """
    Returns a coloured string stating if a prediction was correct or incorrect
    :param pred_idx: The predicted index
    :param ans_idx: The correct index
    :return: a coloured string of the prediction result
    """
    if pred_idx == ans_idx:
        return "\033[32mCorrect\033[0m"
    else:
        return "\033[31mIncorrect\033[0m"
