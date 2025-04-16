import numpy as np

from layer import Layer


# inherit from base class Layer
class FCLayer(Layer):
    # input_size = number of input neurons
    # output_size = number of output neurons
    def __init__(self, input_size, output_size):
        self.weights = (
            np.random.rand(input_size, output_size) - 0.5
        )  # So the weights and biases are randomized
        self.bias = np.random.rand(1, output_size) - 0.5

    # returns output for a given input
    def forward_propagation(self, input):
        self.input = input
        self.output = (
            np.dot(self.input, self.weights) + self.bias
        )  # So this is the output(input) i gues the dot product of the input and weights + the bias, makes sense
        return self.output

    # computes dE/dW, dE/dB for a given output_error=dE/dY. Returns input_error=dE/dX.
    def backward_propagation(self, output_error, learning_rate):
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)
        # dBias = output_error

        # update parameters
        self.weights -= (
            learning_rate * weights_error
        )  # Makes sense the weight becomes itself minus the learning weigt times weight error I guess this is nabla E or so
        self.bias -= learning_rate * output_error
        return input_error


# activation function and its derivative
def tanh(x):
    return np.tanh(x)


def tanh_prime(x):
    return 1 - np.tanh(x) ** 2


def relu(x):
    return np.maximum(0, x)


def relu_prime(x):
    return np.where(x > 0, 1, 0)


class SoftmaxLayer(Layer):
    def forward_propagation(self, input):
        # Compute the softmax output
        exp_values = np.exp(input - np.max(input, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return probabilities

    def backward_propagation(self, output_error, learning_rate):
        # Compute the gradient of the softmax function
        return (
            output_error  # No gradient update needed for softmax layer in this context
        )
