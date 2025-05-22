import numpy as np
import fir_conv

def log_edge(
    in_img_array: np.ndarray
) -> np.ndarray:
    """
    Performs Laplacian of Gaussian (LoG) edge detection on a grayscale image.
    Uses a pre-defined 5x5 LoG kernel and detects zero-crossings.

    Args:
        in_img_array: 2D numpy array representing the grayscale input image.

    Returns:
        A 2D numpy array representing the binary edge image (0 or 255).
    """
    #5x5 LoG kernel
    log_kernel = np.array([
        [0,  0, -1,  0,  0],
        [0, -1, -2, -1,  0],
        [-1,-2, 16, -2, -1],
        [0, -1, -2, -1,  0],
        [0,  0, -1,  0,  0]
    ], dtype=float)

    # Origin for the 5x5 LoG mask is its center
    mask_origin = np.array([2, 2])

    # Convolve the image with the LoG kernel
    log_convolved_image, _ = fir_conv.fir_conv(in_img_array, log_kernel, mask_origin=mask_origin)

    # Zero-crossing detection 4-connectivity
    rows, cols = log_convolved_image.shape
    edge_image = np.zeros_like(log_convolved_image, dtype=np.uint8)

    # Iterate over the interior pixels (to allow neighbor checks)
    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            pixel = log_convolved_image[r, c]
            neighbors = [
                log_convolved_image[r - 1, c],  # North
                log_convolved_image[r + 1, c],  # South
                log_convolved_image[r, c - 1],  # West
                log_convolved_image[r, c + 1]   # East
            ]

            is_zero_crossing = False
            if pixel > 0:
                if any(n < 0 for n in neighbors):
                    is_zero_crossing = True
            elif pixel < 0:
                if any(n > 0 for n in neighbors):
                    is_zero_crossing = True
            
            if is_zero_crossing:
                edge_image[r, c] = 1

    return edge_image