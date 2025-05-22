import numpy as np
from typing import Tuple, Optional

def fir_conv(
    in_img_array: np.ndarray,
    h: np.ndarray,
    in_origin: Optional[np.ndarray] = None,
    mask_origin: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Performs 2D FIR convolution on a grayscale image.

    Args:
        in_img_array: 2D numpy array representing the grayscale input image.
        h: 2D numpy array representing the convolution mask (kernel).
        in_origin: Optional 1D numpy array [row, col] specifying the origin of the input image.
                   If None, the origin is assumed to be the center.
        mask_origin: Optional 1D numpy array [row, col] specifying the origin of the mask.
                     If None, the origin is assumed to be the center.

    Returns:
        A tuple containing:
        - out_img_array (np.ndarray): The convolved image.
        - out_origin (Optional[np.ndarray]): The origin of the output image. 
          This is the same as `in_origin` if both `in_origin` and `mask_origin` 
          were explicitly provided by the user; otherwise, it's None.
    """
    img_rows, img_cols = in_img_array.shape
    mask_rows, mask_cols = h.shape

    # Store whether origins were explicitly provided by the user
    user_provided_in_origin = in_origin is not None
    user_provided_mask_origin = mask_origin is not None

    # Determine effective origins for computation (default to center if not provided)
    eff_in_origin = in_origin if user_provided_in_origin else np.array([img_rows // 2, img_cols // 2])
    eff_mask_origin = mask_origin if user_provided_mask_origin else np.array([mask_rows // 2, mask_cols // 2])

    # Calculate padding needed based on the effective mask origin
    pad_top = eff_mask_origin[0]
    pad_bottom = mask_rows - 1 - eff_mask_origin[0]
    pad_left = eff_mask_origin[1]
    pad_right = mask_cols - 1 - eff_mask_origin[1]

    # Create padded image with zeros
    padded_img = np.pad(
        in_img_array, 
        ((pad_top, pad_bottom), (pad_left, pad_right)), 
        mode='constant', 
        constant_values=0
    )

    # Initialize output image
    out_img_array = np.zeros_like(in_img_array, dtype=float) # Output should be float for precision

    # Perform convolution
    # Iterate over the original image dimensions
    for i in range(img_rows):
        for j in range(img_cols):
            # Determine the ROI in the padded image based on the effective mask origin
            roi_start_row = i + pad_top - eff_mask_origin[0]
            roi_end_row = roi_start_row + mask_rows
            roi_start_col = j + pad_left - eff_mask_origin[1]
            roi_end_col = roi_start_col + mask_cols
            
            roi = padded_img[roi_start_row:roi_end_row, roi_start_col:roi_end_col]
            
            # Perform element-wise multiplication and sum
            if roi.shape == h.shape:
                out_img_array[i, j] = np.sum(roi * h)

    # Determine the output origin
    out_origin: Optional[np.ndarray] = None
    if user_provided_in_origin and user_provided_mask_origin:
        out_origin = in_origin # Use the original user-provided in_origin

    return out_img_array, out_origin
