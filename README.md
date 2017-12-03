# README

FYS2010 - Digital Image Processing
Home Exam 04.03.2016 - 18.03.2016

### Run:

To run a problem:

  ```Shell
  python problem.py # e.g. python 1A.py
  ```

### Algorithms:

Functions implemented in spatial_filters.py:
- 2-dimensional spatial convolution
- Mean filter (arithmetic and geometric)
- Median filter
- Adaptive median filter
- Adaptive local noise reduction filter

Functions implemented in freq_filters.py:
- Lowpass and highpass filter (Ideal, Butterworth and Gaussian)
- Bandreject and bandpass filter (Ideal, Butterworth and Gaussian)
- Notchreject and notchpass filter (Ideal, Butterworth and Gaussian)
- Filtering procedure for frequency domain

Functions implemented in scaling_functions.py:
- im2double, scaling function to [0,1]
- im2uint8, scaling function to [0,255]
