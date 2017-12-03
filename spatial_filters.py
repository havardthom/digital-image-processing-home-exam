import numpy as np

# 2-Dimensional Convolution in spatial domain
###############################################################################
# INPUT
# f:		   Input image
# w:		   Input filter
###############################################################################
# OUTPUT
# result:	   Output image
###############################################################################
def spatial_convolution2d(f, w):
	# Get filter size
	m, n = w.shape

	# Pad impulse image
	f = np.pad(f, (m-1, n-1), 'constant', constant_values=0)

	# Get padded f size
	x, y = f.shape

	edgex = m/2
	edgey = n/2

	# Initialize result image
	result = np.zeros_like(f)

	# Rotate filter 180 degrees
	rot_w = np.flipud(np.fliplr(w))

	# Loop through padded f
	for i in range(edgex, x - edgex):
		for j in range(edgey, y - edgey):

			# Loop through filter on each pixel in padded f and convolute
			for s in range(m):
				for t in range(n):
					result[i, j] += rot_w[s,t] * f[i + s - edgex, j + t - edgey]

	# Crop result image back to original size
	result = result[m-1:-(m-1), n-1:-(n-1)]

	return result


# Mean Filter
###############################################################################
# INPUT
# img:			Input image
# s:		  	Shape of filter (default is 3x3)
# ftype:		Filter type ('arithmetic' or 'geometric')
###############################################################################
# OUTPUT
# result:		Output image
###############################################################################
def mean_filter(img, s=3, ftype='geometric'):
	x, y = img.shape
	# Initialize result image
	result = np.zeros_like(img)

	filter_edge = s/2

	# Traverse through image
	for i in range(0,x):
		for j in range(0,y):

		 	 if ftype == 'arithmetic':
 				sum_values = 0
 			 elif ftype=='geometric':
				product_values = 1

			 count = 0

			 # Traverse through filter
			 for u in range(s):
				 for v in range(s):
					 # Get current position
					 cur_x = (i + u - filter_edge)
					 cur_y = (j + v - filter_edge)

					 # Stay inside image boundaries
					 if((cur_x >= 0) and (cur_y >= 0) and (cur_x < x) and (cur_y < y)):
					 	 if ftype == 'arithmetic':
 							 # Get sum of values
 							 sum_values += img[cur_x, cur_y]
						 elif ftype=='geometric':
							 # Get product of values
							 product_values *= img[cur_x, cur_y]


						 count+=1

			 if ftype == 'arithmetic':
				 # Get arithmetic mean value
				 mean = sum_values/count
			 elif ftype=='geometric':
				 # Get geometric mean value
				 mean = product_values**(1.0/count)
				#  print mean

			 # Round off to closest integer
			 result[i, j] = mean

	return result

# Median Filter
###############################################################################
# INPUT
# img:			Input image
# s:		  	Shape of filter (default is 3x3)
###############################################################################
# OUTPUT
# result:		Output image
###############################################################################
def median_filter(img, s=3):

	x, y = img.shape
	# Initialize result image
	result = np.zeros_like(img)

	# Traverse through image
	for i in range(0, x):
		for j in range(0, y):

			 # Create new filter list
			 filtr = []
			 filter_edge = s/2

			 # Traverse through filter
			 for u in range(s):
				 for v in range(s):
					 # Get current position
					 cur_x = (i + u - filter_edge)
					 cur_y = (j + v - filter_edge)

					 # Stay inside image boundaries
					 if((cur_x >= 0) and (cur_y >= 0) and (cur_x < x) and (cur_y < y)):
						 # Append value to filter list
						 filtr.append(img[cur_x, cur_y])

			 # Convert filter list to numpy array
			 filtr = np.asarray(filtr)
			 # Output median value in filter region
			 result[i, j] = np.median(filtr)

	return result



# Adaptive Median Filter
###############################################################################
# INPUT
# img:			Input image
# s:		  	Start shape of filter (default is 3x3)
# s_max:		Maximum shape of filter (default is 7x7)
###############################################################################
# OUTPUT
# result:		Output image
###############################################################################
def adaptive_median_filter(img, s=3, s_max=7):

	x, y = img.shape
	# Initialize result image
	result = np.zeros_like(img)

	# Traverse through image
	for i in range(0, x):
		for j in range(0, y):
			 # Set current filter size to starting filter size
			 s_cur = s
			 # While current filter size is smaller or equal to maximum filter size
			 while s_cur <= s_max:
				 # Create new filter list
				 filtr = []

				 filter_edge = s_cur/2

				 # Traverse through filter
				 for u in range(s_cur):
					 for v in range(s_cur):
						 # Get current position
						 cur_x = (i + u - filter_edge)
						 cur_y = (j + v - filter_edge)

						 # Stay inside image boundaries
						 if((cur_x >= 0) and (cur_y >= 0) and (cur_x < x) and (cur_y < y)):
							 # Append value to filter list
							 filtr.append(img[cur_x, cur_y])

							 # Get value in center of filter region
							 if cur_x == i and cur_y == j:
								 z_xy = filtr[-1]

				 # Convert filter list to numpy array
				 filtr = np.asarray(filtr)
				 # Get minimum value in filter region
				 z_min = np.amin(filtr)
				 # Get maximum value in filter region
				 z_max = np.amax(filtr)
				 # Get median value in filter region
				 z_med = np.median(filtr)

				 # If z_med is not an impulse: check next case. else: increase window size
				 if z_min < z_med < z_max:
					 # If z_xy is not an impulse: output z_xy. else: output z_med
					 if z_min < z_xy < z_max:
						 result[i, j] = z_xy
					 else:
						 result[i, j] = z_med
					 # Break to exit while loop
					 break
				 else:
					 s_cur += 2

			 else:
				 result[i, j] = z_med # Output median value if maximum window size has been surpassed

	return result

# Adaptive Local Noise Reduction Filter
###############################################################################
# INPUT
# img:			Input image
# var_g:		Estimate of overall noise variance in image
# s:			Shape of filter (default is 3x3)
###############################################################################
# OUTPUT
# result:		Output image
###############################################################################
def adaptive_lnr_filter(img, var_g, s=3):

	x, y = img.shape
	# Initialize result image
	result = np.zeros_like(img)

	filter_edge = s/2

	# Traverse through image
	for i in range(0,x):
		for j in range(0,y):
			 # Create new filter list
			 filtr = []

			 # Traverse through filter
			 for u in range(s):
				 for v in range(s):
					 # Get current position
					 cur_x = (i + u - filter_edge)
					 cur_y = (j + v - filter_edge)

					 # Stay inside image boundaries
					 if((cur_x >= 0) and (cur_y >= 0) and (cur_x < x) and (cur_y < y)):
						 # Append value to filter list
						 filtr.append(img[cur_x, cur_y])

			 # Convert filter list to numpy array
			 filtr = np.array(filtr)
			 # Get local mean from filter
			 mean_l = np.mean(filtr)
			 # Get local variance from filter
			 var_l = np.var(filtr)

			 # If local variance is smaller than global variance, set ratio to 1
			 if var_g <= var_l:
				 r = var_g / var_l
			 else:
				 r = 1

			 # Get the output value and round off to nearest integer
			 result[i, j] = img[i, j] - (r * (img[i, j] - mean_l))

	return result
