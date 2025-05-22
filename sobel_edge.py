import fir_conv
import numpy as np

def sobel_edge(
    in_img_array: np.ndarray,
    thres: float
) -> np.ndarray:
    """
    Performs Sobel edge detection on a grayscale image.

    Args:
        in_img_array: 2D numpy array representing the grayscale input image.
        thres: positive float corresponding to the threshold value of the gradient magnitude, 
        above which an edge.

    Returns:
        A 2D numpy array representing the binary edge image (0 or 255).
    """
    # Sobel operators (kernels)
    Gx = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=float)

    Gy = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1,-2,-1]
    ], dtype=float)

    # Origin for the 3x3 Sobel masks is their center
    mask_origin = np.array([1, 1])

    # Convolve with Gx to get the horizontal gradient component
    g1, _ = fir_conv.fir_conv(in_img_array, Gx, mask_origin=mask_origin) 

    # Convolve with Gy to get the vertical gradient component
    g2, _ = fir_conv.fir_conv(in_img_array, Gy, mask_origin=mask_origin)

    # Calculate the gradient magnitude
    gradient_magnitude = np.sqrt(g1**2 + g2**2)

    # Normalize gradient magnitude to 0-255 for consistent thresholding (optional but good practice)
    # However, the threshold is absolute, so direct application is fine.
    # Since in_img_array values are in [0,1], gradient_magnitude will be scaled accordingly.
    # For Sobel 3x3 kernels, if input is [0,1], g1/g2 are roughly in [-4, 4],
    # so gradient_magnitude can be up to np.sqrt(4**2 + 4**2) = np.sqrt(32) approx 5.65.
    # The 'thres' value should be chosen with this scale in mind.
    
    # Apply threshold to get the binary edge image
    # Pixels with gradient magnitude > thres are edges (255), else non-edges (0)
    edge_image = np.zeros_like(gradient_magnitude, dtype=np.uint8)
    edge_image[gradient_magnitude > thres] = 255

    return edge_image
