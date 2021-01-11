import numpy as np
from neural_network.activation_functions import sigmoid, sigmoid_derivative


class SimpleNeuralNetwork:
    def __init__(self, x, y, n_output=1, lr=0.01):
        self.loss = []
        self.y = y.reshape(-1, 1)
        self.x = x
        self.lr = lr
        self.weights_1 = np.random.random((x.shape[1], 5))
        self.weights_2 = np.random.random((5, n_output))
        self.output = np.zeros(n_output)

    def compute_loss(self):
        return np.sum(np.power(self.output - self.y, 2), axis=0)

    def _backpropagation(self):

        delta_2 = (self.y - self.output) * sigmoid_derivative(np.matmul(self.layer_1, self.weights_2))
        gradient_2 = 2 * np.dot(self.layer_1.T, delta_2)

        delta_1 = np.dot(delta_2, self.weights_2.T) * sigmoid_derivative(np.matmul(self.x, self.weights_1))
        gradient_1 = 2 * np.dot(self.x.T, delta_1)

        self.weights_1 += self.lr * gradient_1
        self.weights_2 += self.lr * gradient_2

    def _forwardpropagation(self):
        self.layer_1 = sigmoid(np.matmul(self.x, self.weights_1))
        self.output = sigmoid(np.matmul(self.layer_1, self.weights_2))

    def fit(self, iterations=100, print_error=True):
        self.loss = []
        for i in range(0, iterations):
            self._forwardpropagation()
            self.loss.append(self.compute_loss())
            self._backpropagation()
            if print_error and i % 100 == 0:
                print('Epoch: {}, error: {}'.format(i + 1, self.compute_loss()))

    def predict(self, new_data):
        layer_1 = sigmoid(np.dot(new_data, self.weights_1))
        predictions = sigmoid(np.dot(layer_1, self.weights_2))
        return predictions


class NeuralNetwork:
    def __init__(self, x, y,  n_output=1, lr=0.01):
        self.loss = []
        self.y = y.reshape(-1, 1)
        self.x = x
        self.lr = lr
        self.weights_1 = np.random.normal(size=(x.shape[1], 16))
        self.weights_2 = np.random.normal(size=(16, 8))
        self.weights_3 = np.random.normal(size=(8, n_output))
        self.output = np.zeros(n_output)

    def compute_loss(self):
        return np.sum(np.power(self.output - self.y, 2), axis=0)

    def _backpropagation(self):
        delta_3 = (self.y - self.output) * sigmoid_derivative(np.matmul(self.layer_2, self.weights_3))
        gradient_3 = 2 * np.dot(self.layer_2.T, delta_3)

        delta_2 = np.dot(delta_3, self.weights_3.T) * sigmoid_derivative(np.matmul(self.layer_1, self.weights_2))
        gradient_2 = 2 * np.dot(self.layer_1.T, delta_2)

        delta_1 = np.dot(delta_2, self.weights_2.T) * sigmoid_derivative(np.matmul(self.x, self.weights_1))
        gradient_1 = 2 * np.dot(self.x.T, delta_1)

        self.weights_1 += self.lr * gradient_1
        self.weights_2 += self.lr * gradient_2
        self.weights_3 += self.lr * gradient_3

    def _forward_propagation(self):
        self.layer_1 = sigmoid(np.matmul(self.x, self.weights_1))
        self.layer_2 = sigmoid(np.matmul(self.layer_1, self.weights_2))
        self.output = sigmoid(np.matmul(self.layer_2, self.weights_3))

    def fit(self, iterations=100, print_error=True):
        self.loss = []
        for i in range(0, iterations):
            self._forward_propagation()
            self.loss.append(self.compute_loss())
            self._backpropagation()
            if print_error and i % 100 == 0:
                print('Epoch: {}, error: {}'.format(i + 1, self.compute_loss()))

    def predict(self, new_data):
        layer_1 = sigmoid(np.dot(new_data, self.weights_1))
        layer_2 = sigmoid(np.dot(layer_1, self.weights_2))
        predictions = sigmoid(np.dot(layer_2, self.weights_3))
        # predictions = np.where(predictions > 0.5, 1, 0)
        return predictions
