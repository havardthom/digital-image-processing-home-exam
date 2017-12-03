import numpy as np

# Lowpass Filter
###############################################################################
# INPUT
# shape:		P x Q shape of filter
# d0:		  	Cutoff frequency
# ftype:		Filter type ('ideal', 'butterworth' or 'gaussian')
# n:			Order of filter (only applies for butterworth filters)
# u_k:			u position of notch pair (only applies for notch filters)
# v_k:			v position of notch pair (only applies for notch filters)
###############################################################################
# OUTPUT
# H:		   A lowpass filter with input parameters
###############################################################################
def lowpass_filter(shape, d0=160, ftype='butterworth', n=2, u_k=0, v_k=0):

	P, Q = shape
	# Initialize filter with zeros
	H = np.zeros((P, Q))

	# Traverse through filter
	for u in range(0, P):
		for v in range(0, Q):
			# Get euclidean distance from point D(u,v) to the center
			D_uv = np.sqrt((u - P/2 + u_k)**2 + (v - Q/2 + v_k)**2)

			# Define lowpass transfer funtion according to filter type
			if ftype == 'ideal':

				if D_uv <= d0:
					H[u, v] = 1.0

			elif ftype == 'butterworth':

				H[u, v] = 1/(1 + (D_uv/d0)**(2*n))

			elif ftype == 'gaussian':

				H[u, v] = np.exp(-D_uv**2 / (2*d0)**2)

	return H

# Highpass Filter
###############################################################################
# INPUT
# shape:		P x Q shape of filter
# d0:		  	Cutoff frequency
# ftype:		Filter type ('ideal', 'butterworth' or 'gaussian')
# n:			Order of filter (only applies for butterworth filters)
# u_k:			u position of notch pair (only applies for notch filters)
# v_k:			v position of notch pair (only applies for notch filters)
###############################################################################
# OUTPUT
# H:		   A highpass filter with input parameters
###############################################################################
def highpass_filter(shape, d0=160, ftype='butterworth', n=2, u_k=0, v_k=0):
	# Inverse of lowpass
	H = 1.0 - lowpass_filter(shape, d0, ftype, n, u_k, v_k)
	return H

# Bandreject Filter
###############################################################################
# INPUT
# shape:		P x Q shape of filter
# d0:		  	Cutoff frequency
# w:			Width of the band
# ftype:		Filter type ('ideal', 'butterworth' or 'gaussian')
# n:			Order of filter (only applies for butterworth filters)
###############################################################################
# OUTPUT
# H:		   A bandreject filter with input parameters
###############################################################################
def bandreject_filter(shape, d0=160, w=20, ftype='butterworth', n=2):

	P, Q = shape
	# Initialize filter with ones
	H = np.ones((P, Q))

	# Traverse through filter
	for u in range(0, P):
		for v in range(0, Q):
			# Get euclidean distance from point D(u,v) to the center
			D_uv = np.sqrt((u - (P/2))**2 + (v - (Q/2))**2)

			# Define bandreject transfer funtion for each filter type
			if ftype == 'ideal':

				if (d0 - (w/2)) <= D_uv <= (d0 + (w/2)):
					H[u, v] = 0.0

			elif ftype == 'butterworth':

				if D_uv == d0: # To avoid dividing by zero
					H[u, v] = 0
				else:
					H[u, v] = 1/(1 + ((D_uv*w)/(D_uv**2 - d0**2))**(2*n))

			elif ftype == 'gaussian':

				if D_uv == 0: # To avoid dividing by zero
					H[u, v] = 1
				else:
					H[u, v] = 1.0 - np.exp(-((D_uv**2 - d0**2) / (D_uv * w))**2)

	return H

# Bandpass Filter
###############################################################################
# INPUT
# shape:		P x Q shape of filter
# d0:		  	Cutoff frequency
# w:			Width of the band
# ftype:		Filter type ('ideal', 'butterworth' or 'gaussian')
# n:			Order of filter (only applies for butterworth filters)
###############################################################################
# OUTPUT
# H:		   A bandpass filter with input parameters
###############################################################################
def bandpass_filter(shape, d0=160, w=20, ftype='butterworth', n=2):
	# Inverse of bandreject
	H = 1.0 - bandreject_filter(shape, d0, w, ftype, n)
	return H

# Notch Reject Filter (several notch pairs not implemented)
###############################################################################
# INPUT
# shape:		P x Q shape of filter
# d0:		  	Cutoff frequency
# ftype:		Filter type ('ideal', 'butterworth' or 'gaussian')
# n:			Order of filter (only applies for butterworth filters)
# u_k:			u position of notch pair
# v_k:			v position of notch pair
###############################################################################
# OUTPUT
# H:		   A notch reject filter with input parameters
###############################################################################
def notch_reject_filter(shape, d0=160, ftype='butterworth', n=2, u_k=0, v_k=0):
	# Form product of highpass filters at position (-u_k, -v_k) and (u_k, v_k)
	H = highpass_filter(shape, d0, ftype, n, -u_k, -v_k) * highpass_filter(shape, d0, ftype, n, u_k, v_k)
	return H

# Notch Pass Filter (several notch pairs not implemented)
###############################################################################
# INPUT
# shape:		P x Q shape of filter
# d0:		  	Cutoff frequency
# ftype:		Filter type ('ideal', 'butterworth' or 'gaussian')
# n:			Order of filter (only applies for butterworth filters)
# u_k:			u position of notch pair
# v_k:			v position of notch pair
###############################################################################
# OUTPUT
# H:		   A notch pass filter with input parameters
###############################################################################
def notch_pass_filter(shape, d0=160, ftype='butterworth', n=2, u_k=0, v_k=0):
	# Inverse of notch reject
	H = 1.0 - notch_reject_filter(shape, d0, ftype, n, u_k, v_k)
	return H



# Filter image in frequency domain
###############################################################################
# INPUT
# img:			Input image
# fclass:		Filter class ('lowpass', 'highpass', 'bandreject' or 'bandpass')
# ftype:		Filter type ('ideal', 'butterworth' or 'gaussian')
# d0:			Cutoff frequency
# w:			Width of the band (only applies for bandpass/bandreject filters)
# n:			Order of filter (only applies for butterworth filters)
###############################################################################
# OUTPUT
# G:		   Output image
# H:		   Filter image
# P:		   Power spectrum of input image
###############################################################################
def filter_image_freq(img, fclass='lowpass', ftype='butterworth', d0=160, w=20, n=2, u_k=0, v_k=0):

	# Get padding parameters
	M, N = img.shape
	P = 2*M
	Q = 2*N

	# Take the fourier transform of the image, with padding to shape P X Q
	F = np.fft.fft2(img, s=(P,Q))

	# Shift the low frequencies to the center.
	F = np.fft.fftshift(F)

	# Get power spectrum of the image
	pow_spec = np.abs(F)**2

	# Create a filter with input parameters
	if fclass == 'lowpass':
		H = lowpass_filter(F.shape, d0, ftype, n)
	elif fclass == 'highpass':
		H = highpass_filter(F.shape, d0, ftype, n)
	elif fclass == 'bandreject':
		H = bandreject_filter(F.shape, d0, w, ftype, n)
	elif fclass == 'bandpass':
		H = bandpass_filter(F.shape, d0, w, ftype, n)
	elif fclass == 'notchreject':
		H = notch_reject_filter(F.shape, d0, ftype, n, u_k, v_k)
	elif fclass == 'notchpass':
		H = notch_pass_filter(F.shape, d0, ftype, n, u_k, v_k)

	# Form product of image with filter
	G = F * H

	# Shift frequencies back
	G = np.fft.ifftshift(G)

	# Inverse fourier transform to get output image in spatial domain
	G = np.fft.ifft2(G)

	# Get real values
	G = np.abs(G)

	# Extract M x N image from top left quadrant
	G = G[0:M, 0:N]

	# Return output image, the filter used and the power spectrum of input image
	return G, np.abs(H), np.log(pow_spec)
