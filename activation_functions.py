import numpy as np

def relu(x):
    """
    Rectified Linear Unit (ReLU) activation function
    :param x: Initial data
    :return: Data after transformation
    """
    return (x > 0) * x
def relu_derivative(x):
    """
    Rectified Linear Unit (ReLU) activation function derivative
    :param x: Initial data
    :return: Data after transformation
    """
    return x > 0

def sigmoid(x):
    """
    Sigmoid activation function
    :param x: Initial data
    :return: Data after transformation
    """
    return 1 / (1 + np.exp(-x))
def sigmoid_derivative(x):
    """
    Sigmoid activation function derivative
    :param x: Initial NumPy object
    :return: NumPy object after transformation
    """
    sig = sigmoid(x)
    return sig * (1 - sig)

def tanh(x):
    """
    Hyperbolic tangent activation function
    :param x: Initial NumPy object
    :return: NumPy object after transformation
    """
    return np.tanh(x)
def tanh_derivative(x):
    """
    Hyperbolic tangent activation function derivative
    :param x: Initial data
    :return: Data after transformation
    """
    return 1 - (tanh(x) ** 2)