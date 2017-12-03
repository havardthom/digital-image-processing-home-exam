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
	img = mpimg.imread('data/P3_fig3.png')

	# Filter image in frequency domain using a notch reject filter of order 2
	# with cutoff frequency 15 at position (150,150) and (-150,-150)
	G, H, P = filter_image_freq(img, fclass='notchreject', ftype='butterworth',
								d0=15, n=2, u_k=150, v_k=150)

	# Get spatial noise pattern using the notch pass filter with same parameters
	G2, _, _ = filter_image_freq(img, fclass='notchpass', ftype='butterworth',
								 d0=15, n=2, u_k=150, v_k=150)

	# Scale to uint8 before displaying
	img = im2uint8(img)
	G = im2uint8(G)
	G2 = im2uint8(G2)

	# Plot results
	fig = plt.figure()
	fig.suptitle('3F: Denoising', fontsize=20)

	ax = plt.subplot(2,3,1)
	ax.set_title("Original Image")
	plt.imshow(img, cmap=plt.cm.gray, vmin=min_r, vmax=max_r)
	plt.axis('off')

	ax = plt.subplot(2,3,3)
	ax.set_title("Result")
	plt.imshow(G, cmap=plt.cm.gray, vmin=min_r, vmax=max_r)
	plt.axis('off')

	ax = plt.subplot(2,3,4)
	ax.set_title("Image Power Spectrum")
	plt.imshow(P, cmap=plt.cm.gray)
	plt.axis('off')

	ax = plt.subplot(2,3,5)
	ax.set_title("Notch Reject Filter (d0=15, n=2, u_k=150, v_k=150)")
	plt.imshow(H, cmap=plt.cm.gray)
	plt.axis('off')

	ax = plt.subplot(2,3,6)
	ax.set_title("Spatial Noise Pattern")
	plt.imshow(G2, cmap=plt.cm.gray)
	plt.axis('off')

	# figManager = plt.get_current_fig_manager()
	# figManager.window.showMaximized()
	plt.show()
