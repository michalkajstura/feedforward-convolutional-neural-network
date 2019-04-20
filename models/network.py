import numpy as np
import feedforward_cnn.models.layers as layers


class FeedforwardNetwork:
    def __init__(self, params):
        self.params = params

    def predict(self, x):
        output = self.forward(x)
        return np.argmax(output, axis=1)

    def forward(self, x):
        # x is 36 x 36 x 1
        x = layers.convolution(x, self.params[0]['weight'], self.params[0]['bias'], padding=1)  # 36 x 36 x 16
        x = layers.relu(x)
        x = layers.batchnorm(x, self.params[1]['gamma'], self.params[1]['beta'],
                             self.params[1]['running_mean'], self.params[1]['running_var'])
        x = layers.convolution(x, self.params[2]['weight'], self.params[2]['bias'], padding=1)  # 36 x 36 x 32
        x = x.reshape(-1, 36 * 36 * 32)
        x = layers.linear_forward(x, self.params[3]['weight'], self.params[3]['bias'])
        x = layers.relu(x)
        x = layers.linear_forward(x, self.params[4]['weight'], self.params[4]['bias'])
        return x
