import numpy as np

#Rectified Linear Unit (ReLU) activation function
def relu(x):
    return (x > 0) * x
def relu_derivative(x):
    return x > 0

#Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoid_derivative(x):
    sig = sigmoid(x)
    return sig * (1 - sig)

def tanh(x):
    return np.tanh(x)
def tanh_derivative(x):
    return 1 - (tanh(x) ** 2)