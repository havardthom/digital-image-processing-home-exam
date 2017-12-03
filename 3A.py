import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from freq_filters import filter_image_freq
from scaling_functions import im2double, im2uint8

# Global max and min intensity values for plotting
max_r = np.iinfo(np.uint8).max
min_r = np.iinfo(np.uint8).min

if __name__ == "__main__":
	# Load images to numpy arrays
	img1 = mpimg.imread('data/P3_fig1.png')
	img2 = mpimg.imread('data/P3_fig2.png')
	img3 = mpimg.imread('data/P3_fig3.png')


	# Get power spectrum of image 3
	_, _, P3 = filter_image_freq(img3)


	# Scale to uint8 before displaying
	img1 = im2uint8(img1)
	img2 = im2uint8(img2)
	img3 = im2uint8(img3)

	# Extract a subimage from image 1 with reasonably constant background intensity
	subimage1 = img1[240:290, 100:200]

	# Extract a subimage from image 2 with reasonably constant background intensity
	subimage2 = img2[240:290, 100:200]

	# Plot results
	fig = plt.figure()
	fig.suptitle('3A: Denoising', fontsize=20)

	ax = plt.subplot(2,3,1)
	ax.set_title("Image 1")
	plt.imshow(img1, cmap=plt.cm.gray, vmin=min_r, vmax=max_r)
	plt.axis('off')

	ax = plt.subplot(2,3,2)
	ax.set_title("Image 2")
	plt.imshow(img2, cmap=plt.cm.gray, vmin=min_r, vmax=max_r)
	plt.axis('off')

	ax = plt.subplot(2,3,3)
	ax.set_title("Image 3")
	plt.imshow(img3, cmap=plt.cm.gray, vmin=min_r, vmax=max_r)
	plt.axis('off')

	ax = plt.subplot(2,3,4)
	ax.set_title("Subimage 1 Histogram")
	plt.hist(subimage1.ravel(), bins=max_r, range=(min_r, max_r), normed=True, color='k')
	plt.xlim((min_r,max_r))

	ax = plt.subplot(2,3,5)
	ax.set_title("Subimage 2 Histogram")
	plt.hist(subimage2.ravel(), bins=max_r, range=(min_r, max_r), normed=True, color='k')
	plt.xlim((min_r,max_r))

	ax = plt.subplot(2,3,6)
	ax.set_title("Image 3 Power Spectrum")
	plt.imshow(P3, cmap=plt.cm.gray)
	plt.axis('off')

	plt.show()
