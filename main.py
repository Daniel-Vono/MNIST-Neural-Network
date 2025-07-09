import os

import numpy as np

import file_utils
import utils
import activation_functions

# The side length of the square image
IMG_LEN = 28

# The side length of the kernel and stride for the convolutional layer
KERNEL_LEN = 3
STRIDE = 1

# The shape of the fully connected layers of the network
INPUT_SIZE = 784
LAYER_1_SIZE = 96
OUTPUT_SIZE = 10

# The alpha, batch size, and dropout probability used during training
ALPHA = 0.001
BATCH_SIZE = 8
DROPOUT_CHANCE = 0.25

# The name of the files where weights are stored
WEIGHTS_0_1_FILE_NAME = 'weights_0_1.npy'
WEIGHTS_1_2_FILE_NAME = 'weights_1_2.npy'
KERNEL_FILE_NAME = 'kernel.npy'

# File paths to the training data
train_images_path = 'dataset/train-images.idx3-ubyte'
train_labels_path = 'dataset/train-labels.idx1-ubyte'

# Loads the dataset
train_images = file_utils.load_mnist_images(train_images_path) # Shape is (60000, 28, 28)
train_labels = file_utils.load_mnist_labels(train_labels_path) # Shape is (60000,)

# Initialize the weights for the kernel and load an existing kernel if it exists
kernel = (np.random.random((KERNEL_LEN, KERNEL_LEN)) - 0.5) * 0.5
if os.path.exists(KERNEL_FILE_NAME):
    kernel = np.load(KERNEL_FILE_NAME)

# Initialize the weights for layer 0_1 and load any existing weights for the layer if they exist
weights_0_1 = np.random.random((INPUT_SIZE, LAYER_1_SIZE)) - 0.5
if os.path.exists(WEIGHTS_0_1_FILE_NAME):
    weights_0_1 = np.load(WEIGHTS_0_1_FILE_NAME)

# Initialize the weights for layer 1_2 and load any existing weights for the layer if they exist
weights_1_2 = np.random.random((LAYER_1_SIZE, OUTPUT_SIZE)) - 0.5
if os.path.exists(WEIGHTS_1_2_FILE_NAME):
    weights_1_2 = np.load(WEIGHTS_1_2_FILE_NAME)

#Store sum gradients for mini-batch learning
batch_gradient_1_2 = 0
batch_gradient_0_1 = 0
batch_gradient_kernel = np.zeros(kernel.shape)

#Stores data about the current state of the mini-batch
curr_batch_processing_num = 1
batch_epoch = 0

#Stores the number of correct predictions made by the network
num_correct = 0

# Trains the network
for epoch in range (6):
    for iteration in range(train_labels.size):
        '''
        FORWARD PROPAGATION
        '''

        # Pad the image for convolution
        padded_img = np.zeros([IMG_LEN + 2, IMG_LEN + 2])
        padded_img[1:IMG_LEN + 1, 1:IMG_LEN + 1] = train_images[iteration] / 255.0

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

        # Apply dropout to layer 1
        dropout_vector = np.random.choice([0, 1], size=(1, LAYER_1_SIZE), p=[DROPOUT_CHANCE, 1 - DROPOUT_CHANCE])
        layer_1 *= dropout_vector / (1 - DROPOUT_CHANCE)

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
        goal_pred[0][train_labels[iteration]] = 1.0

        '''
        BACK PROPAGATION
        '''

        # Update batch epoch
        batch_epoch += 1

        # Calculate cross entropy error for output
        delta_output = probabilities - goal_pred
        error_output = - np.log(probabilities[0][train_labels[iteration]])

        # Update weights_1_2 batch gradient
        weighted_delta_output = np.dot(layer_1.T, delta_output)  # Scale each delta by the input that caused it. The transverse causes the resulting matrix shape to be the same as the weights
        batch_gradient_1_2 += weighted_delta_output

        # Chain rule the derivative between the hidden layer to output layer by multiplying by the delta of the input layer to hidden layer
        delta_layer_0_1 = np.dot(delta_output, weights_1_2.T) * activation_functions.tanh_derivative(layer_1)
        delta_layer_0_1 *= dropout_vector
        batch_gradient_0_1 += layer_0.T.dot(delta_layer_0_1)

        # Chain rule del layer 1 del kernel with the activation function of the convolution and previous derivatives for the fully connected layer
        delta_con_0 = np.dot(weights_0_1, delta_layer_0_1.T)
        delta_con_0 = delta_con_0.T
        delta_con_0 = delta_con_0.reshape(IMG_LEN, IMG_LEN) * activation_functions.tanh_derivative(convoluted_img)
        delta_kernel = np.zeros(kernel.shape)

        #Calculates the sum loss of the kernel for each pixel it produced
        for row in range(IMG_LEN):
            for col in range(IMG_LEN):
                #Get the original batch used at this point in the convolution
                region = padded_img[row:row + KERNEL_LEN, col:col + KERNEL_LEN]

                #Multiply the region matrix by the delta with respect to what the convolution produced for that pixel
                delta_kernel += region * delta_con_0[row, col]

        # Updates the kernel sum gradient
        batch_gradient_kernel += delta_kernel

        # Updates the weights if the batch is complete
        if batch_epoch == BATCH_SIZE:
            # Updates the weights based on the average gradient of the batch
            weights_1_2 = weights_1_2 - (ALPHA * batch_gradient_1_2 / BATCH_SIZE)
            weights_0_1 = weights_0_1 - (ALPHA * batch_gradient_0_1 / BATCH_SIZE)
            kernel = kernel - (ALPHA * batch_gradient_kernel / BATCH_SIZE)

            # Increment the number of batches processed
            curr_batch_processing_num += 1

            # Resets the sum gradients and batch epoch
            batch_gradient_1_2 = 0
            batch_gradient_0_1 = 0
            batch_gradient_kernel = np.zeros(kernel.shape)
            batch_epoch = 0

        # Increment the number of correct predictions if the model predicted correctly
        if prediction_idx == train_labels[iteration]:
            num_correct = num_correct + 1

        # Prints the current information on the state of the training
        print("Accuracy:", 100 * num_correct / (iteration + 1 + (epoch * train_labels.size)), "Batch:", curr_batch_processing_num, "Epoch Number:", epoch)

# Saves all weights to files
np.save(KERNEL_FILE_NAME, kernel)
np.save(WEIGHTS_0_1_FILE_NAME, weights_0_1)
np.save(WEIGHTS_1_2_FILE_NAME, weights_1_2)