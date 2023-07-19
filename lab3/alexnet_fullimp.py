# TensorFlow and tf.keras
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import scipy
import numpy as np
import cv2
import time

# create conv2d_time, conv2d_fft, maxpool, and dense layers

# first convolutional layer: input of 3x227x227 convolved with 96 3x11x11 kernels with stride 4
# inputs is height, width, channels
# kernels is height, width, input channels, output channels (translates to: for each output channels there are weights for height, width, and channel)
def FFTConv2D(inputs, kernels, biases, stride=1, padding=0, mode='valid', activation='none'):
    input_c, input_h, input_w = inputs.shape
    output_c, _, kernel_h, kernel_w  = kernels.shape
    # Output height = (Input height + padding height top + padding height bottom - kernel height) / (stride height) + 1
    output_h = (input_h + padding*2 - kernel_h) // stride + 1
    output_w = (input_w + padding*2 - kernel_w) // stride + 1

    # if padding
    if(padding > 0):
        inputs = np.pad(inputs, pad_width=((0,0), (padding,padding), (padding, padding)), mode='constant', constant_values=0)

    # get the output set up
    output = np.zeros(shape=(output_c, output_h, output_w))

    for out_c in range(output_c):
        for in_c in range(input_c):
            output[out_c,:,:] += scipy.signal.fftconvolve(inputs[in_c,:,:], kernels[out_c, in_c,:,:], mode=mode)[::stride, ::stride]
        output[out_c,:,:] += biases[out_c]

    # apply relu
    if(activation == 'relu'):
        output = np.maximum(0, output)

    return output

def Conv2D(inputs, kernels, biases, stride=1, padding=0, mode='valid', activation='none'):
    input_c, input_h, input_w = inputs.shape
    output_c, _, kernel_h, kernel_w  = kernels.shape
    # Output height = (Input height + padding height top + padding height bottom - kernel height) / (stride height) + 1
    output_h = (input_h + padding*2 - kernel_h) // stride + 1
    output_w = (input_w + padding*2 - kernel_w) // stride + 1

    # if padding
    if(padding > 0):
        inputs = np.pad(inputs, pad_width=((0,0), (padding,padding), (padding, padding)), mode='constant', constant_values=0)

    # get the output set up
    output = np.zeros(shape=(output_c, output_h, output_w))

    for out_c in range(output_c):
        for in_c in range(input_c):
            output[out_c,:,:] += scipy.signal.convolve2d(inputs[in_c,:,:], kernels[out_c, in_c,:,:], mode=mode)[::stride, ::stride]
        output[out_c,:,:] += biases[out_c]

    # apply relu
    if(activation == 'relu'):
        output = np.maximum(0, output)

    return output

def MaxPooling(inputs, kernel_size, stride):
    input_c, input_h, input_w = inputs.shape

    # Output height = (Input height + padding height top + padding height bottom - kernel height) / (stride height) + 1
    output_h = (input_h - kernel_size) // stride + 1
    output_w = (input_w - kernel_size) // stride + 1

    output = np.zeros(shape=(input_c, output_h, output_w))

    # iterate through each item in output
    for out_h in range(output_h):
        for out_w in range(output_w):
            # get respective values in input
            h = out_h * stride
            w = out_h * stride
            # get 3d chunk of input
            inputs_chunk = inputs[:, h:h + kernel_size, w:w + kernel_size]
            output[:, out_h, out_w] = np.amax(inputs_chunk, axis=(1, 2))

    return output

def FullyConnected(inputs, weights, biases, activation='none'):
    if(mode == 'relu'):
        return np.maximum(0, np.dot(inputs, weights) + biases)
    else:
        return np.dot(inputs, weights) + biases

def Flatten(inputs):
    return inputs.flatten()

inputs = np.zeros(shape=(3,227,227))

# load the weights and biases for all of the layers
conv1_b = np.load('./assets_alexnet/cnn_data/alexnet_model/conv1_b.npy')
conv1_w = np.load('./assets_alexnet/cnn_data/alexnet_model/conv1_w.npy')
conv2_b = np.load('./assets_alexnet/cnn_data/alexnet_model/conv2_b.npy')
conv2_w = np.load('./assets_alexnet/cnn_data/alexnet_model/conv2_w.npy')
conv3_b = np.load('./assets_alexnet/cnn_data/alexnet_model/conv3_b.npy')
conv3_w = np.load('./assets_alexnet/cnn_data/alexnet_model/conv3_w.npy')
conv4_b = np.load('./assets_alexnet/cnn_data/alexnet_model/conv4_b.npy')
conv4_w = np.load('./assets_alexnet/cnn_data/alexnet_model/conv4_w.npy')
conv5_b = np.load('./assets_alexnet/cnn_data/alexnet_model/conv5_b.npy')
conv5_w = np.load('./assets_alexnet/cnn_data/alexnet_model/conv5_w.npy')
fc6_b = np.load('./assets_alexnet/cnn_data/alexnet_model/fc6_b.npy')
fc6_w = np.load('./assets_alexnet/cnn_data/alexnet_model/fc6_w.npy')
fc7_b = np.load('./assets_alexnet/cnn_data/alexnet_model/fc7_b.npy')
fc7_w = np.load('./assets_alexnet/cnn_data/alexnet_model/fc7_w.npy')
fc8_b = np.load('./assets_alexnet/cnn_data/alexnet_model/fc8_b.npy')
fc8_w = np.load('./assets_alexnet/cnn_data/alexnet_model/fc8_w.npy')

# read the labels from a text file
with open('./labels.txt') as f:
    labels = f.readlines()

labels = [x.replace('"', '') for x in labels]
labels = [x.replace('\n', '') for x in labels]

# read an image to run inference on
im = cv2.imread('dog2.png')
im = im.reshape((3,227,227))


start = time.perf_counter()
outFFT = FFTConv2D(im, conv1_w, conv1_b, stride=4, activation='relu')
print(f"FFTConv2D took {time.perf_counter()-start} seconds")

start = time.perf_counter()
outDirect = Conv2D(im, conv1_w, conv1_b, stride=4, activation='relu')
print(f"Conv2D took {time.perf_counter()-start} seconds")

avDiff = np.average(outFFT - outDirect)
avWeights = np.average(conv1_w)
minWeight = np.min(conv1_w)

print(f"The average difference in output tensor is {avDiff}. This is {avDiff/avWeights}% of the average of all weights and {avDiff/minWeight}% of the smallest weight.")


# building alexnet
out = FFTConv2D(im, conv1_w, conv1_b, stride=4, activation='relu')
out = MaxPooling(out, 3, 2)
out = Conv2D(out, conv2_w, conv2_b, stride=1, padding=2, activation='relu')
out = MaxPooling(out, 3, 2)
out = Conv2D(out, conv3_w, conv3_b, stride=1, padding=1, activation='relu')
out = Conv2D(out, conv4_w, conv4_b, stride=1, padding=1, activation='relu')
out = Conv2D(out, conv5_w, conv5_b, stride=1, padding=1, activation='relu')
out = MaxPooling(out, 3, 2)
out = Flatten(out)
out = FullyConnected(out, fc6_w, fc6_b, activation='relu')
out = FullyConnected(out, fc7_w, fc7_b, activation='relu')
out = FullyConnected(out, fc8_w, fc8_b, activation='relu')

guess = np.argmax(out)
print(labels[guess])
