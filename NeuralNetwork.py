import numpy as np


class NeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size):
        self.weights = []
        self.biases = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            self.weights.append(np.random.randn(prev_size, hidden_size))
            self.biases.append(np.zeros((1, hidden_size)))
            prev_size = hidden_size
        self.weights.append(np.random.randn(prev_size, output_size))
        self.biases.append(np.zeros((1, output_size)))

    def forward(self, x):
        hidden_layer = x
        for weight, bias in zip(self.weights[:-1], self.biases[:-1]):
            hidden_layer = np.dot(hidden_layer, weight) + bias
            hidden_layer = np.maximum(0, hidden_layer)  # ReLU activation function
        output_layer = np.dot(hidden_layer, self.weights[-1]) + self.biases[-1]
        return output_layer

    def train(self, input_data, output_data, epochs, learning_rate):
        for epoch in range(epochs):
            # Forward pass
            predictions = self.forward(input_data)

            # Compute the loss
            loss = np.mean(np.square(predictions - output_data))

            # Backward pass
            delta = 2 * (predictions - output_data)
            for i in reversed(range(len(self.weights))):
                weight_gradient = np.dot(self.weights[i].T, delta)
                bias_gradient = np.sum(delta, axis=0)

                # Update weights and biases
                self.weights[i] -= learning_rate * weight_gradient
                self.biases[i] -= learning_rate * bias_gradient

                delta = np.dot(delta, self.weights[i].T)

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss}")
