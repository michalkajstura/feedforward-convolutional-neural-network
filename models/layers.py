import numpy as np


def convolution(x, filters, bias, stride=1, padding=0):
    """
    :param x: Input matrix. N x C x H x W
    :param filters: Filters / kernels / weights matrix. F_out x C x F_h x F_w
    :param bias: Vector of bias for each filter. F_out
    :param stride: Size of convolution step.
    :param padding: Zero padding around image
    :return: Activations map. N x F_out x H_out x W_out
    """
    n, _, x_h, x_w = x.shape
    f, _, filter_size_h, filter_size_w = filters.shape

    h_out = 1 + (x_h + 2 * padding - filter_size_h) // stride
    w_out = 1 + (x_w + 2 * padding - filter_size_w) // stride

    out = np.zeros((n, f, h_out, w_out))
    x_padded = np.pad(x, ((0,), (0,), (padding,), (padding,)), 'constant')
    f_reshaped = filters.reshape(f, -1).T  # F_out x 1

    for h in range(0, h_out):
        for w in range(0, w_out):
            h_start, w_start = h * stride, w * stride
            h_end, w_end = h_start + filter_size_h, w_start + filter_size_w
            out[:, :, h, w] = np.dot(
                x_padded[:, :, h_start: h_end, w_start: w_end].reshape(n, -1),
                f_reshaped
            ) + bias

    return out


def linear_forward(x, w, b):
    return x.dot(w.T) + b


def relu(x):
    return np.maximum(0, x)


def cross_entropy_loss(x, labels):
    n = labels.shape[0]
    log_softmax = -np.log(np.exp(x[np.arange(n), labels]) / np.sum(np.exp(x), axis=1))
    return np.mean(log_softmax)


def batchnorm(x, gamma, beta, running_mean, running_var, eps=1e-5):
    N, C, H, W = x.shape
    x_normalized = x - running_mean.reshape((1, C, 1, 1))
    x_normalized /= np.sqrt(running_var.reshape((1, C, 1, 1)) + eps)
    return x_normalized * gamma.reshape((1, C, 1, 1)) + beta.reshape((1, C, 1, 1))

