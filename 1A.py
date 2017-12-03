import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Global max and min intensity values for plotting
max_r = np.iinfo(np.uint8).max
min_r = np.iinfo(np.uint8).min

if __name__ == "__main__":
	# Load image to numpy array
	img = mpimg.imread('data/Fig0310(b)(washed_out_pollen_image).tif')

	fig = plt.figure()
	fig.suptitle('1A: Histogram', fontsize=20)

	# Plot image
	ax = plt.subplot(1,2,1)
	ax.set_title("Image")
	plt.imshow(img, cmap=plt.cm.gray, vmin=min_r, vmax=max_r)
	plt.axis('off')

	# Plot histogram
	ax = plt.subplot(1,2,2)
	ax.set_title("Image Histogram")
	plt.hist(img.ravel(), bins=max_r, range=(min_r, max_r), normed=True, color='k')
	plt.xlim((min_r,max_r))

	plt.show()
