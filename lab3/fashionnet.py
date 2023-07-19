import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

from fashionnet_utils import FashionNet

fashion_mnist = keras.datasets.fashion_mnist
(train_val_images, train_val_labels), (test_images, test_labels) = fashion_mnist.load_data()

#preprocess the data
split = 50000
#split into validation and normal training
validation_images = train_val_images[split:]
train_images = train_val_images[:split]

validation_labels = train_val_labels[split:]
train_labels = train_val_labels[:split]

train_images = train_images / 255.0
validation_images = validation_images / 255.0
test_images = test_images / 255.0

# reshape data
train_images = train_images[..., np.newaxis]
validation_images = validation_images[..., np.newaxis]
test_images = test_images[..., np.newaxis]

labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

model = FashionNet(test_images=test_images, test_labels=test_labels, num_images=1000)

accuracy, total_time, inf_time = model.run("direct")
print(f"Direct Convolution Model Accuracy: {(accuracy) * 100}")
print(f"Total Model Run Time: {total_time} s")
print(f"Average Single Image Inference Time: {inf_time * 1000} ms \n")

accuracy, total_time, inf_time = model.run("hybrid1")
print(f"Hybrid1 Convolution Model Accuracy: {(accuracy) * 100}")
print(f"Total Model Run Time: {total_time} s")
print(f"Average Single Image Inference Time: {inf_time * 1000} ms \n")

accuracy, total_time, inf_time = model.run("hybrid2")
print(f"Hybrid2 Convolution Model Accuracy: {(accuracy) * 100}")
print(f"Total Model Run Time: {total_time} s")
print(f"Average Single Image Inference Time: {inf_time * 1000} ms \n")

accuracy, total_time, inf_time = model.run("fft")
print(f"FFT Convolution Model Accuracy: {(accuracy) * 100}")
print(f"Total Model Run Time: {total_time} s")
print(f"Average Single Image Inference Time: {inf_time * 1000} ms \n")
