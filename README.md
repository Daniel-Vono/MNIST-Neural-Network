# MNIST-Neural-Network

This is a neural network made using only Python and NumPy which takes 28x28 gray scale images of drawn digits as inputs, and can predict what number it is.

### Architecture and Training Methods
1. Convolutional layer 
   - 3x3 kernel
   - Stride of 1
   - Zero padding
   - tanh activation function
   - Flattened to a vector of length 784
2. Fully connected layer of size 96
   - tanh activation function
   - dropout applied
3. Fully connected layer of size 10 (10 possible outputs)
   - Softmax used to calculate probability distribution