import numpy as np
import feedforward_cnn.models.layers as layers


class FeedforwardNetwork:
    def __init__(self, weights, biases):
        assert len(weights) == len(biases)
        self.weights = weights
        self.biases = biases

    def predict(self, x):
        output = self.forward(x)
        return np.argmax(output, axis=1)

    def forward(self, x):
        # x is 36 x 36 x 1
        x = layers.convolution(x, self.weights[0], self.biases[0], padding=1)  # 36 x 36 x 16
        x = layers.relu(x)
        x = x.reshape(-1, 36 * 36 * 16)
        x = layers.linear_forward(x, self.weights[1], self.biases[1])
        x = layers.relu(x)
        x = layers.linear_forward(x, self.weights[2], self.biases[2])
        return x
