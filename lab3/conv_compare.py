import cv2
import numpy as np
import scipy
import time

img = cv2.imread('irongiant.png', 0)
fft_data = []
direct_data = []
top_k_size = 30 + 1

for kernel_size in range(3, top_k_size, 3):
    filter = np.ones(shape=(kernel_size, kernel_size)) / (kernel_size * kernel_size)

    # Run FFT Convolution
    fft_start = time.time()

    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20*np.log(np.abs(fshift))
    g = np.fft.fft2(filter, fshift.shape)
    F_gaussian = np.fft.fftshift(g)
    F_filtered_img = fshift*F_gaussian
    filtered_img = np.fft.ifft2(np.fft.ifftshift(F_filtered_img)).real

    fft_data.append(time.time() - fft_start)
    print(f"Single layer FFT Convolution ({kernel_size}x{kernel_size} kernel): Process took {fft_data[-1]} seconds.")

    # Run direct convolution
    direct_start = time.time()
    x = scipy.signal.convolve(img, filter, method='direct', mode='same')
    
    direct_data.append(time.time() - direct_start)
    print(f"Single layer Direct Convolution ({kernel_size}x{kernel_size} kernel): Process took {direct_data[-1]} seconds.")

print(direct_data)
print(fft_data)

import matplotlib.pyplot as plt

plt.plot(list(range(3, top_k_size, 3)), direct_data, "-co", label="Direct Convolution Time")
plt.plot(list(range(3, top_k_size, 3)), fft_data, "--bo", label="FFT Convolution Time")
plt.title("Direct vs FFT Convolution with Variable Kernel Sizes")
plt.xlabel("Kernel Size")
plt.ylabel("Layer Execution Time (s)")
plt.legend()
plt.show()
