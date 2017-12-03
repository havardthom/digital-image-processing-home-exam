import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from spatial_filters import spatial_convolution2d
from scaling_functions import im2double, im2uint8

# Global max and min intensity values for plotting
max_r = np.iinfo(np.uint8).max
min_r = np.iinfo(np.uint8).min

if __name__ == "__main__":
	# Load image
	img = mpimg.imread('data/Fig0343(a)(skeleton_orig).tif')

	# Initialize laplacian filter
	l = np.negative(np.ones((3,3)))
	l[1, 1] = 8

	# Convert image to double format with intensities scaled between [0, 1]
	img_scaled = im2double(img)

	# Get laplacian image by convoluting the image with the laplacian filter
	lap = spatial_convolution2d(img_scaled, l)

	# Get sharpened image
	sharpimg_scaled = img_scaled + lap

	# Scale laplacian image to [0, 1] intensity range
	lap_scaled = im2double(lap)

	# Scale to uint8 before displaying
	sharpimg = im2uint8(sharpimg_scaled)
	lap = im2uint8(lap_scaled)

	fig = plt.figure()
	fig.suptitle('2BC: Spatial Filtering', fontsize=20)

	ax = plt.subplot(1,3,1)
	ax.set_title("Original Image")
	plt.imshow(img, cmap=plt.cm.gray, vmin=min_r, vmax=max_r)
	plt.axis('off')

	ax = plt.subplot(1,3,2)
	ax.set_title("Laplacian Image")
	plt.imshow(lap, cmap=plt.cm.gray, vmin=min_r, vmax=max_r)
	plt.axis('off')

	ax = plt.subplot(1,3,3)
	ax.set_title("Laplacian Sharpened Image")
	plt.imshow(sharpimg, cmap=plt.cm.gray, vmin=min_r, vmax=max_r)
	plt.axis('off')

	# figManager = plt.get_current_fig_manager()
	# figManager.window.showMaximized()
	plt.show()
