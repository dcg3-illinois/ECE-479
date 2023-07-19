import numpy as np
import scipy

import time
class FashionNet:
    def __init__(self, test_images: np.array, test_labels: np.array, num_images: int) -> None:
        if (num_images > len(test_images)):
            raise Exception("Number of images cannot be greater than number of test images.")
        
        self.test_images = test_images
        self.test_labels = test_labels
        # self.labels = labels # if needed, can add classification function
        self.num_images = num_images
        
        data_path = "./fashionnet/"
        self.fnconv1_w = np.load(data_path + "fnconv1_w.npy")
        self.fnconv2_w = np.load(data_path + "fnconv2_w.npy")
        self.fnconv1_b = np.load(data_path + "fnconv1_b.npy")
        self.fnconv2_b = np.load(data_path + "fnconv2_b.npy")
        self.fc1_w = np.load(data_path + "fc1_w.npy")
        self.fc1_b = np.load(data_path + "fc1_b.npy")
        self.fc2_w = np.load(data_path + "fc2_w.npy")
        self.fc2_b = np.load(data_path + "fc2_b.npy")
        self.fc3_w = np.load(data_path + "fc3_w.npy")
        self.fc3_b = np.load(data_path + "fc3_b.npy")
        
    def run(self, type: str="direct") -> tuple:
        if (type != "direct" and type != "hybrid1" and type != "hybrid2" and type != "fft"):
            raise Exception("Invalid model type chosen. Please choose 'direct', 'hybrid1', 'hybrid2', or 'fft'.")
        
        count = 0
        total_inf_time = time.time()
        inf_times = []
        
        for i in range(self.num_images):
            first_im = self.test_images[i]

            single_inf_time = time.time()
            
            # building fashionnet
            if type == "direct" or type == "hybrid2":
                out = self.Conv2D("direct", first_im, self.fnconv1_w, self.fnconv1_b, stride=1, padding=0, activation='relu', mode='same')
            else:
                out = self.Conv2D("fft", first_im, self.fnconv1_w, self.fnconv1_b, stride=1, padding=0, activation='relu', mode='same')

            out = self.MaxPooling(out, 2, 2)
            
            if type == "direct" or type == "hybrid1":
                out = self.Conv2D("direct", out, self.fnconv2_w, self.fnconv2_b, stride=1, activation='relu', mode='valid')
            else: 
                out = self.Conv2D("fft", out, self.fnconv2_w, self.fnconv2_b, stride=1, activation='relu', mode='valid')
            
            out = self.MaxPooling(out, 2, 2)
            out = self.Flatten(out)
            out = self.FullyConnected(out, self.fc1_w, self.fc1_b, activation='relu')
            out = self.FullyConnected(out, self.fc2_w, self.fc2_b, activation='relu')
            out = self.FullyConnected(out, self.fc3_w, self.fc3_b, activation='softmax')
            inf_times.append(time.time() - single_inf_time)
            
            guess = np.argmax(out)
            if (guess == self.test_labels[i]): count += 1
            
        return (count / self.num_images, time.time() - total_inf_time, sum(inf_times) / len(inf_times))

    def Conv2D(self, type, inputs, kernels, biases, stride=1, padding=0, mode='valid', activation='none'):
        input_h, input_w, input_c = inputs.shape
        kernel_h, kernel_w, _, output_c = kernels.shape
        # Output height = (Input height + padding height top + padding height bottom - kernel height) / (stride height) + 1
        if (mode != 'same'):
            output_h = (input_h + padding*2 - kernel_h) // stride + 1
            output_w = (input_w + padding*2 - kernel_w) // stride + 1
        else:
            output_h = input_h // stride
            output_w = input_w // stride

        # if padding
        if (padding > 0):
            inputs = np.pad(inputs, pad_width=((padding,padding), (padding, padding), (0,0)), mode='constant', constant_values=0)

        # get the output set up
        output = np.zeros(shape=(output_h, output_w, output_c))
        
        # Perform convolution using scipy
        for k in range(output_c):
            kernel = kernels[ :, :, :, k]
            kernel = np.rot90(kernel, 2, axes=(0, 1))
            for c in range(input_c):
                if type == "direct":
                    output[ :, :, k] += scipy.signal.convolve2d(inputs[ :, :, c], kernel[:, :, c], mode=mode)[::stride, ::stride]
                else:
                    output[ :, :, k] += scipy.signal.fftconvolve(inputs[ :, :, c], kernel[:, :, c], mode=mode)[::stride, ::stride]
            output[ :, :, k] += biases[k]

        # apply relu
        if (activation == 'relu'):
            output = np.maximum(0, output)

        return output

    def MaxPooling(self, inputs, kernel_size, stride):
        h, w, num_filters = inputs.shape
        output = np.zeros((h // 2, w // 2, num_filters))

        for im_region, i, j in self.iterate_regions(inputs):
            output[i, j] = np.amax(im_region, axis=(0, 1))

        return output

    def FullyConnected(self, inputs, weights, biases, activation='none'):
        if (activation == 'relu'):
            return np.maximum(0, inputs @ weights + biases)
        elif (activation=='softmax'):
            return scipy.special.softmax(inputs @ weights + biases)
        else:
            return inputs @ weights + biases

    def Flatten(self, inputs):
        return inputs.flatten()
    
    def iterate_regions(self, image):
        h, w, _ = image.shape
        new_h = h // 2
        new_w = w // 2

        for i in range(new_h):
            for j in range(new_w):
                im_region = image[(i * 2):(i * 2 + 2), (j * 2):(j * 2 + 2)]
                yield im_region, i, j