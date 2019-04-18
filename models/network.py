import numpy as np

class FeedforwardNetwork:
    def __init__(self, weights, biases):
        assert len(weights) == len(biases)
        self.weights = weights
        self.biases = biases



    @staticmethod
    def linear_forward(x, w, b):
        return x.dot(w.T) + b

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def cross_entropy_loss(x, labels):
        n = labels.shape[0]
        log_softmax = -np.log(np.exp(x[np.arange(n), labels]) / np.sum(np.exp(x), axis=1))
        return np.mean(log_softmax)

    @staticmethod
    def predict(output):
        return np.argmax(output, axis=1)

    @staticmethod
    def batchnorm(x, gamma, beta, running_stats, mode='test', eps=1e-5, momentum=0.9):
        N, D = x.shape
        running_mean = running_stats.get('running_mean', np.zeros(D, dtype=x.dtype))
        running_var = running_stats.get('running_var', np.zeros(D, dtype=x.dtype))

        if mode == 'train':
            sample_mean = x.mean(axis=0)
            sample_var = np.sum((x - sample_mean) ** 2, axis=0) / N
            running_mean = momentum * running_mean + (1 - momentum) * sample_mean
            running_var = momentum * running_var + (1 - momentum) * sample_var
            x_hat = (x - sample_mean) / np.sqrt(sample_var + eps)
        elif mode == 'train':
            x_hat = (x - running_mean) / np.sqrt(running_var + eps)
        else:
            raise ValueError('Invalid mode: %s' % mode)

        running_stats['running_mean'] = running_mean
        running_stats['running_var'] = running_var

        return x_hat * gamma + beta

    @staticmethod
    def convolution(x, filters, bias, stride=1, padding=0):
        N, _, H, W = x.shape
        F, _, HH, WW = filters.shape

        h_prim = int(1 + (H + 2 * padding - HH) / stride)
        w_prim = int(1 + (W + 2 * padding - WW) / stride)
        out = np.zeros((N, F, h_prim, w_prim))
        x_pad = np.pad(x, ((0,), (0,), (padding,), (padding,)), 'constant')

        for n in range(N):
            for f in range(F):
                for h_out in range(h_prim):
                    for w_out in range(w_prim):
                        h_start = h_out * stride
                        h_end = h_start + HH
                        w_start = w_out * stride
                        w_end = w_start + WW
                        kernels = filters[f]
                        convolved = \
                            FeedforwardNetwork.convolve(x_pad[n, :, h_start: h_end, w_start: w_end],
                                                        kernels) + bias[f]

                        out[n, f, h_out, w_out] = convolved

        return out

    @staticmethod
    def convolve(arr, kernel):
        return np.sum(np.multiply(arr, kernel))

