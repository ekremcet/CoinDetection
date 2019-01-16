import numpy as np
from utils import *


def nanargmax(arr):
    # Returns the index of max non-nan value
    idx = np.nanargmax(arr)
    index = np.unravel_index(idx, arr.shape)

    return index


def conv_backprop(prev_layer, input, filter, stride):
    (f_count, f_channel, f_dim, _) = filter.shape
    (_, i_dim, _) = input.shape

    # Initialize derivatives
    d_output = np.zeros(input.shape)
    d_filter = np.zeros(filter.shape)
    d_bias = np.zeros((f_count, 1))

    for curr_f in range(f_count):
        curr_y = out_y = 0
        while curr_y + f_dim <= i_dim:
            curr_x = out_x = 0
            while curr_x + f_dim <= i_dim:
                # loss gradient of filter (used to update the filter)
                d_filter[curr_f] += prev_layer[curr_f, out_y, out_x] * input[:, curr_y:curr_y + f_dim, curr_x:curr_x + f_dim]
                # loss gradient of the input to the convolution operation (conv1 in the case of this network)
                d_output[:, curr_y:curr_y + f_dim, curr_x:curr_x + f_dim] += prev_layer[curr_f, out_y, out_x] * filter[curr_f]
                curr_x += stride
                out_x += 1
            curr_y += stride
            out_y += 1
        # loss gradient of the bias
        d_bias[curr_f] = np.sum(prev_layer[curr_f])

    return d_output, d_filter, d_bias


def maxpool_backprop(prev_pool, input, f, s):
    (i_channel, i_dim, _) = input.shape

    d_output = np.zeros(input.shape)

    for curr_c in range(i_channel):
        curr_y = out_y = 0
        while curr_y + f <= i_dim:
            curr_x = out_x = 0
            while curr_x + f <= i_dim:
                # Find largest value in current window
                (a, b) = nanargmax(input[curr_c, curr_y:curr_y + f, curr_x:curr_x + f])
                d_output[curr_c, curr_y + a, curr_x + b] = prev_pool[curr_c, out_y, out_x]

                curr_x += s
                out_x += 1
            curr_y += s
            out_y += 1

    return d_output
