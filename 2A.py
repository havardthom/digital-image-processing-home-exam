import numpy as np

from spatial_filters import spatial_convolution2d

if __name__ == "__main__":
    # Initialize impulse image f
    f = np.zeros((5,5))
    f[2][2] = 1

    # Initialize filter
    w = np.arange(1,10).reshape(3,3)

    # Get result
    result = spatial_convolution2d(f, w)
    print result.astype(np.uint8)
