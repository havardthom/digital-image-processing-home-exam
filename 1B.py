import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Global max and min intensity values for plotting
max_r = np.iinfo(np.uint8).max
min_r = np.iinfo(np.uint8).min

if __name__ == "__main__":
	# Load image to numpy array
	img = mpimg.imread('data/Fig0310(b)(washed_out_pollen_image).tif')
	u, v = img.shape

	# Get number of occurences of each intensity level
	n = np.bincount(img.ravel(), minlength=max_r+1).astype(np.float64)

	# Get cumulative distribution function
	su = 0
	s = np.zeros_like(n)
	for j in range(0, max_r+1):
		su += n[j]
		s[j] = su

	s = np.around(s*max_r/img.size)

	# Create output image by mapping intensity values in the input image to s values
	img_eq = np.zeros_like(img)
	for i in range(0, u):
		for j in range(0, v):
			img_eq[i, j] = s[img[i,j]]


	fig = plt.figure()
	fig.suptitle('1B: Histogram Equalization', fontsize=20)

	# Plot equalized image
	ax = plt.subplot(1,2,1)
	ax.set_title("Equalized Image")
	plt.imshow(img_eq, cmap=plt.cm.gray, vmin=min_r, vmax=max_r)
	plt.axis('off')

	# Plot equalized histogram
	ax = plt.subplot(1,2,2)
	ax.set_title("Equalized Histogram")
	plt.hist(img_eq.ravel(), bins=max_r, range=(min_r, max_r), normed=True, color='k')
	plt.xlim((min_r,max_r))

	plt.show()
