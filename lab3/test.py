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

num_images = 1000
f = open("./fashionnet/labels.txt", "a")
for i in range(num_images):
    plt.imsave(f"./fashionnet/images/{i}.png", test_images[i][:, :, 0])
    f.write(f"{i} {test_labels[i]} \n")

f.close()
