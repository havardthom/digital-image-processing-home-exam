import numpy as np

# Convert image to double [0, 1] scale
def im2double(img):
    min_v = np.amin(img)
    max_v = np.amax(img)

    out = (img.astype(np.float64) - min_v) / (max_v - min_v)
    return out

# Convert image to uint8 [0, 255] scale (assumes img is scaled in range [0,1])
def im2uint8(img):
    # Scale to [0, 255] and round off to nearest integer
    out = np.around(255*img)
    # Clip values outside range [0, 255]
    out = np.clip(out, a_min=0, a_max=255)
    return out.astype(np.uint8)
