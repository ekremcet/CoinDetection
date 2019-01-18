from layers import *
import numpy as np
from PIL import Image
import os


def convert_label(label):
    return {
        "Chinese": 0,
        "Ottoman": 1,
        "Roman"  : 2
    }[label]


def read_data(folder):
    data = []
    labels = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            img = np.array(Image.open(os.path.join(root, file)))
            img = np.reshape(img, (256 * 256, 3))
            data.append(img)
            labels.append(convert_label(root.split("/")[3]))

    return np.array(data).astype(np.float32), np.array(labels).astype(np.int64)


def initialize_filter(size, scale=1.0):
    # Initialize each filter with mean=0 stddev=1
    stddev = scale / np.sqrt(np.prod(size))
    return np.random.normal(loc=0, scale=stddev, size=size)


def initialize_weight(size):
    # Initialize weights with a random normal distribution
    return np.random.standard_normal(size=size) * 0.01


def predict(image, f1, f2, w3, w4, w5, b1, b2, b3, b4, b5, conv_stride=1, pool_dim=2, pool_stride=2):
    # Forward Pass
    image = image.reshape(3, 256, 256)
    conv1 = convolution(image, f1, b1, conv_stride)  # Conv
    conv1[conv1 <= 0] = 0  # ReLU

    pool1 = maxpool(conv1, pool_dim, pool_stride)  # Pool

    conv2 = convolution(pool1, f2, b2, conv_stride)
    conv2[conv2 <= 0] = 0
    pool2 = maxpool(conv2, pool_dim, pool_stride)

    (nf2, dim2, _) = pool2.shape
    flatten = pool2.reshape((nf2 * dim2 * dim2, 1))  # Flatten the final pool layer

    fc1 = w3.dot(flatten) + b3  # Fully connected layer 1
    fc1[fc1 <= 0] = 0

    fc2 = w4.dot(fc1) + b4  # Fully connected layer 2
    fc2[fc2 <= 0] = 0

    out = w5.dot(fc2) + b5  # Prediction layer
    preds = softmax(out)

    return np.argmax(preds), np.max(preds)
