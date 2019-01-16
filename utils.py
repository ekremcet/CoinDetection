from layers import *
import numpy as np


def read_data(filename, num_images, IMAGE_WIDTH):
    pass


def initialize_filter(size, scale=1.0):
    # Initialize each filter with mean=0 stddev=1
    stddev = scale / np.sqrt(np.prod(size))
    return np.random.normal(loc=0, scale=stddev, size=size)


def initialize_weight(size):
    # Initialize weights with a random normal distribution
    return np.random.standard_normal(size=size) * 0.01


def predict(image, f1, f2, w3, w4, b1, b2, b3, b4, conv_s=1, pool_f=2, pool_s=2):
    '''
    Make predictions with trained filters/weights.
    '''
    conv1 = convolution(image, f1, b1, conv_s)  # convolution operation
    conv1[conv1 <= 0] = 0  # relu activation

    conv2 = convolution(conv1, f2, b2, conv_s)  # second convolution operation
    conv2[conv2 <= 0] = 0  # pass through ReLU non-linearity

    pooled = maxpool(conv2, pool_f, pool_s)  # maxpooling operation
    (nf2, dim2, _) = pooled.shape
    fc = pooled.reshape((nf2 * dim2 * dim2, 1))  # flatten pooled layer

    z = w3.dot(fc) + b3  # first dense layer
    z[z <= 0] = 0  # pass through ReLU non-linearity

    out = w4.dot(z) + b4  # second dense layer
    probs = softmax(out)  # predict class probabilities with the softmax activation function

    return np.argmax(probs), np.max(probs)
