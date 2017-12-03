import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from spatial_filters import mean_filter, adaptive_lnr_filter
from scaling_functions import im2double, im2uint8

# Global max and min intensity values for plotting
max_r = np.iinfo(np.uint8).max
min_r = np.iinfo(np.uint8).min

if __name__ == "__main__":
	# Load image to numpy array
	img = mpimg.imread('data/P3_fig1.png')

	# Filter image in spatial domain using a arithmetic mean filter with filter size 5x5
	g1 = mean_filter(img, s=5, ftype='arithmetic')

	# Filter image in spatial domain using a geometric mean filter with filter size 5x5
	g2 = mean_filter(img, s=5, ftype='geometric')

	# Extract a subimage with reasonably constant background intensity
	subimage = img[240:290, 100:200]

	# Estimate overall noise variance from subimage
	var_g = np.var(subimage)

	# Filter image in spatial domain using an adaptive local noise reduction filter with filter size 5x5
	g3 = adaptive_lnr_filter(img, var_g, s=5)

	# Scale to uint8 before displaying
	img = im2uint8(img)
	g1 = im2uint8(g1)
	g2 = im2uint8(g2)
	g3 = im2uint8(g3)

	# Plot results
	fig = plt.figure()
	fig.suptitle('3B: Denoising', fontsize=20)

	ax = plt.subplot(2,2,1)
	ax.set_title("Original Image")
	plt.imshow(img, cmap=plt.cm.gray, vmin=min_r, vmax=max_r)
	plt.axis('off')

	ax = plt.subplot(2,2,2)
	ax.set_title("Arithmetic Mean Filter (s=5)")
	plt.imshow(g1, cmap=plt.cm.gray, vmin=min_r, vmax=max_r)
	plt.axis('off')

	ax = plt.subplot(2,2,3)
	ax.set_title("Geometric Mean Filter (s=5)")
	plt.imshow(g2, cmap=plt.cm.gray, vmin=min_r, vmax=max_r)
	plt.axis('off')

	ax = plt.subplot(2,2,4)
	ax.set_title("Adaptive Noise Reduction Filter (s=5)")
	plt.imshow(g3, cmap=plt.cm.gray, vmin=min_r, vmax=max_r)
	plt.axis('off')

	plt.tight_layout()
	plt.show()
