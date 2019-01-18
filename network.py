from backpropagation import *
from utils import *
from sklearn.utils import shuffle
import numpy as np
import pickle


def conv(image, label, params, conv_stride, pool_dim, pool_stride):
    [f1, f2, w3, w4, w5, b1, b2, b3, b4, b5] = params
    # Forward Pass
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

    loss = categoricalCrossEntropy(preds, label)  # Calculate Loss

    # Backpropagation
    d_output = preds - label  # derivative of loss w.r.t. final dense layer output
    d_w5 = d_output.dot(fc2.T)  # loss gradient of final dense layer weights
    d_b5 = np.sum(d_output, axis=1).reshape(b5.shape)  # loss gradient of final dense layer biases

    d_dense2 = w5.T.dot(d_output)  # loss gradient of second dense layer outputs
    d_dense2[fc2 <= 0] = 0  # backpropagate through ReLU
    d_w4 = d_dense2.dot(fc1.T)
    d_b4 = np.sum(d_dense2, axis=1).reshape(b4.shape)

    d_dense = w4.T.dot(d_dense2)  # loss gradient of first dense layer outputs
    d_dense[fc1 <= 0] = 0  # backpropagate through ReLU
    d_w3 = d_dense.dot(flatten.T)
    d_b3 = np.sum(d_dense, axis=1).reshape(b3.shape)

    d_flatten = w3.T.dot(d_dense)  # loss gradients of fully-connected layer (pooling layer)

    d_pool2 = d_flatten.reshape(pool2.shape)  # reshape fully connected into dimensions of pooling layer
    d_conv2 = maxpool_backprop(d_pool2, conv2, pool_dim, pool_stride)
    d_conv2[conv2 <= 0] = 0

    d_pool1, d_f2, d_b2 = conv_backprop(d_conv2, pool1, f2, conv_stride)
    d_conv1 = maxpool_backprop(d_pool1, conv1, pool_dim, pool_stride)
    d_conv1[conv1 <= 0] = 0

    d_input, d_f1, d_b1 = conv_backprop(d_conv1, image, f1, conv_stride)

    gradients = [d_f1, d_f2, d_w3, d_w4, d_w5, d_b1, d_b2, d_b3, d_b4, d_b5]

    return gradients, loss


# Optimization
def SGD(x_batch, y_batch, num_classes, lr, img_dim, i_channel, params, cost):
    [f1, f2, w3, w4, w5, b1, b2, b3, b4, b5] = params

    inputs = x_batch
    inputs = inputs.reshape(len(x_batch), i_channel, img_dim, img_dim)
    labels = y_batch

    cost_ = 0
    batch_size = len(x_batch)

    # initialize gradients and momentum,RMS params
    d_f1 = np.zeros(f1.shape)
    d_f2 = np.zeros(f2.shape)
    d_w3 = np.zeros(w3.shape)
    d_w4 = np.zeros(w4.shape)
    d_w5 = np.zeros(w5.shape)
    d_b1 = np.zeros(b1.shape)
    d_b2 = np.zeros(b2.shape)
    d_b3 = np.zeros(b3.shape)
    d_b4 = np.zeros(b4.shape)
    d_b5 = np.zeros(b5.shape)

    for i in range(batch_size):
        img = inputs[i]
        label = np.eye(num_classes)[int(labels[i])].reshape(num_classes, 1)  # convert label to one-hot

        # Collect Gradients for input image
        grads, loss = conv(img, label, params, conv_stride=1, pool_dim=2, pool_stride=2)
        [df1_, df2_, dw3_, dw4_, dw5_, db1_, db2_, db3_, db4_, db5_] = grads

        d_f1 += df1_
        d_b1 += db1_
        d_f2 += df2_
        d_b2 += db2_
        d_w3 += dw3_
        d_b3 += db3_
        d_w4 += dw4_
        d_b4 += db4_
        d_w5 += dw5_
        d_b5 += db5_

        cost_ += loss

    # Parameter Update
    f1 -= lr * d_f1
    b1 -= lr * d_b1

    f2 -= lr * d_f2
    b2 -= lr * d_b2

    w3 -= lr * d_w3
    b3 -= lr * d_b3

    w4 -= lr * d_w4
    b4 -= lr * d_b4

    w5 -= lr * d_w5
    b5 -= lr * d_b5

    cost_ = cost_ / batch_size
    cost.append(cost_)

    params = [f1, f2, w3, w4, w5, b1, b2, b3, b4, b5]

    return params, cost


# Training network
def train(num_classes=3, lr=0.001, img_dim=256, img_depth=3, conv_size=5, f1_count=6, f2_count=16,
          batch_size=64, num_epochs=50, save_path='./weights.pkl'):

    # training data
    train_data, train_label = read_data("./Coins/TrainData/")

    # Initialize all parameters
    f1, f2, w3, w4, w5 = (f1_count, img_depth, conv_size, conv_size), (f2_count, f1_count, conv_size, conv_size), (120, 16 * 61 * 61), (84, 120), (3, 84)
    f1 = initialize_filter(f1)
    f2 = initialize_filter(f2)
    w3 = initialize_weight(w3)
    w4 = initialize_weight(w4)
    w5 = initialize_weight(w5)

    b1 = np.zeros((f1.shape[0], 1))
    b2 = np.zeros((f2.shape[0], 1))
    b3 = np.zeros((w3.shape[0], 1))
    b4 = np.zeros((w4.shape[0], 1))
    b5 = np.zeros((w5.shape[0], 1))

    params = [f1, f2, w3, w4, w5, b1, b2, b3, b4, b5]

    cost = []
    ep = 0
    for epoch in range(num_epochs):
        train_data, train_label = shuffle(train_data, train_label)
        batch_data = [train_data[k:k + batch_size] for k in range(0, train_data.shape[0], batch_size)]
        batch_label = [train_label[k:k + batch_size] for k in range(0, train_label.shape[0], batch_size)]

        for x_batch, y_batch in zip(batch_data, batch_label):
            params, cost = SGD(x_batch, y_batch, num_classes, lr, img_dim, img_depth, params, cost)
            print("Cost: %.2f" % (cost[-1]))

        to_save = [params, cost]
        with open(save_path, 'wb') as file:
            pickle.dump(to_save, file)

        print("Saved!")
        ep += 1
        print("Epoch: " + str(ep) + " Cost: " + str(cost[-1]))


    to_save = [params, cost]

    with open(save_path, 'wb') as file:
        pickle.dump(to_save, file)

    return cost


train()
