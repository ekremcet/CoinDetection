import numpy as np


def convolution(input, filter, b, stride=1):
    (f_count, f_channel, f_dim, _) = filter.shape
    i_channel, i_dim, _ = input.shape  # image dimensions

    out_dim = int((i_dim - f_dim) / stride) + 1  # Output dimension
    output = np.zeros((f_count, out_dim, out_dim))  # Result matrix

    # convolve the filter over every part of the image, adding the bias at each step.
    for curr_f in range(f_count):
        curr_y = out_y = 0
        while curr_y + f_dim <= i_dim:  # Shift the filter until it exits the image
            curr_x = out_x = 0
            while curr_x + f_dim <= i_dim:
                output[curr_f, out_y, out_x] = np.sum(filter[curr_f] *
                                                      input[:, curr_y:curr_y + f_dim, curr_x:curr_x + f_dim]) + \
                                                      b[curr_f]  # Apply filter
                curr_x += stride
                out_x += 1
            curr_y += stride
            out_y += 1

    return output


def maxpool(image, f_dim=2, stride=2):
    i_channel, i_h, i_w = image.shape

    # calculate output dimensions after the maxpooling operation.
    h = int((i_h - f_dim) / stride) + 1
    w = int((i_w - f_dim) / stride) + 1

    # create a matrix to hold the values of the maxpooling operation.
    output = np.zeros((i_channel, h, w))

    for i in range(i_channel):
        # Slide it across all channels
        curr_y = out_y = 0
        while curr_y + f_dim <= i_h:
            curr_x = out_x = 0
            while curr_x + f_dim <= i_w:
                output[i, out_y, out_x] = np.max(image[i, curr_y:curr_y + f_dim, curr_x:curr_x + f_dim])  # Choose the max val
                curr_x += stride
                out_x += 1
            curr_y += stride
            out_y += 1

    return output


def softmax(preds):
    out = np.exp(preds)
    return out/np.sum(out)


def categoricalCrossEntropy(preds, label):
    return -np.sum(label * np.log(preds))
