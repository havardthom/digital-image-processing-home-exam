import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from spatial_filters import median_filter, adaptive_median_filter
from scaling_functions import im2double, im2uint8

# Global max and min intensity values for plotting
max_r = np.iinfo(np.uint8).max
min_r = np.iinfo(np.uint8).min

if __name__ == "__main__":
	# Load image to numpy array
	img = mpimg.imread('data/P3_fig2.png')

	# Filter image in spatial domain using a standard median filter with
	# filter size 3x3
	g = median_filter(img, s=5)
	# Filter image in spatial domain using an adaptive median filter with
	# starting filter size 3x3 and maximum filter size 5x5
	g2 = adaptive_median_filter(img, s=3, s_max=5)


    # Scale to uint8 before displaying
	img = im2uint8(img)
	g = im2uint8(g)
	g2 = im2uint8(g2)

	# Plot results
	fig = plt.figure()
	fig.suptitle('3E: Denoising', fontsize=20)


	ax = plt.subplot(1,3,1)
	ax.set_title("Original Image")
	plt.imshow(img, cmap=plt.cm.gray, vmin=min_r, vmax=max_r)
	plt.axis('off')

	ax = plt.subplot(1,3,2)
	ax.set_title("Median Filter (s=5)")
	plt.imshow(g, cmap=plt.cm.gray, vmin=min_r, vmax=max_r)
	plt.axis('off')

	ax = plt.subplot(1,3,3)
	ax.set_title("Adaptive Median Filter (s=3, s_max=5)")
	plt.imshow(g2, cmap=plt.cm.gray, vmin=min_r, vmax=max_r)
	plt.axis('off')


	plt.show()
