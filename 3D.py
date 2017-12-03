import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from freq_filters import filter_image_freq
from scaling_functions import im2double, im2uint8

# Global max and min intensity values for plotting
max_r = np.iinfo(np.uint8).max
min_r = np.iinfo(np.uint8).min

if __name__ == "__main__":
	# Load image to numpy array
	img = mpimg.imread('data/P3_fig1.png')

	# Filter image in frequency domain using a butterworth lowpass filter of
	# order 2 with cutoff frequency 160
	G, H, P = filter_image_freq(img, fclass='lowpass', ftype='butterworth', d0=160, n=2)

	# Scale to uint8 before displaying
	img = im2uint8(img)
	G = im2uint8(G)

	# Plot results
	fig = plt.figure()
	fig.suptitle('3D: Denoising', fontsize=20)

	ax = plt.subplot(2,2,1)
	ax.set_title("Original Image")
	plt.imshow(img, cmap=plt.cm.gray, vmin=min_r, vmax=max_r)
	plt.axis('off')

	ax = plt.subplot(2,2,2)
	ax.set_title("Image Power Spectrum")
	plt.imshow(P, cmap=plt.cm.gray)
	plt.axis('off')

	ax = plt.subplot(2,2,3)
	ax.set_title("Butterworth Lowpass Filter (d0=160, n=2)")
	plt.imshow(H, cmap=plt.cm.gray)
	plt.axis('off')

	ax = plt.subplot(2,2,4)
	ax.set_title("Result")
	plt.imshow(G, cmap=plt.cm.gray, vmin=min_r, vmax=max_r)
	plt.axis('off')

	plt.show()
