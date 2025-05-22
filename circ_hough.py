import numpy as np
from typing import Tuple

def circ_hough(
    in_img_array: np.ndarray,
    R_max: float,
    dim: np.ndarray,
    V_min: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detects circular outlines in a binary image using the Hough Transform.

    Args:
        in_img_array: 2D numpy array (binary image with values 0 or 1, dtype=int).
        R_max: Maximum potential radius to search for.
        dim: 1D numpy array of 3 integers [num_a_bins, num_b_bins, num_r_bins].
        V_min: Minimum number of votes a cell in the Hough accumulator must have
               to be considered part of a detected circle.

    Returns:
        A tuple (centers, radii):
        - centers (np.ndarray): Kx2 array containing (x, y) coordinates of detected circle centers.
        - radii (np.ndarray): 1D array of length K containing the radii of detected circles.
                                K is the number of circles detected.
    """
    img_rows, img_cols = in_img_array.shape

    num_a_bins = dim[0]
    num_b_bins = dim[1]
    num_r_bins = dim[2]

    a_step = img_cols / num_a_bins if num_a_bins > 0 else img_cols
    b_step = img_rows / num_b_bins if num_b_bins > 0 else img_rows
    r_step = R_max / num_r_bins if num_r_bins > 0 else R_max

    accumulator = np.zeros((num_b_bins, num_a_bins, num_r_bins), dtype=int)
    # Order: (b_bins, a_bins, r_bins) -> (y_center_bins, x_center_bins, radius_bins)
    accumulator = np.zeros((num_b_bins, num_a_bins, num_r_bins), dtype=int)

    # Define angles for parameterizing circles (equation: x = a + r*cos(theta), y = b + r*sin(theta))
    # Or, for voting: a = x_edge - r*cos(theta), b = y_edge - r*sin(theta)
    thetas = np.deg2rad(np.arange(0, 360, 2)) # 2-degree steps, 180 points for better accuracy

    # Get coordinates of edge pixels (where in_img_array == 1)
    y_edges, x_edges = np.where(in_img_array == 1)

    # Voting process
    for y_edge, x_edge in zip(y_edges, x_edges):
        for r_idx in range(num_r_bins):
            # Calculate the representative radius for this bin (center of the bin)
            current_r = (r_idx + 0.5) * r_step
            if current_r <= 0: # Radius must be positive
                continue

            for theta in thetas:
                # Calculate potential center coordinates (a, b)
                a_center_candidate = x_edge - current_r * np.cos(theta)
                b_center_candidate = y_edge - current_r * np.sin(theta)

                # Convert (a, b) to bin indices in the accumulator
                a_bin_idx = int(np.floor(a_center_candidate / a_step))
                b_bin_idx = int(np.floor(b_center_candidate / b_step))

                # Check if bin indices are within valid bounds
                if (0 <= a_bin_idx < num_a_bins and 
                    0 <= b_bin_idx < num_b_bins):
                    accumulator[b_bin_idx, a_bin_idx, r_idx] += 1
    
    # Extract circles from the accumulator
    detected_centers_list = []
    detected_radii_list = []

    # Find cells with votes >= V_min
    b_peak_indices, a_peak_indices, r_peak_indices = np.where(accumulator >= V_min)

    for b_idx, a_idx, r_idx in zip(b_peak_indices, a_peak_indices, r_peak_indices):
        # Convert bin indices back to approximate real-world values (center of bin)
        center_x = (a_idx + 0.5) * a_step
        center_y = (b_idx + 0.5) * b_step
        radius = (r_idx + 0.5) * r_step

        detected_centers_list.append([center_x, center_y])
        detected_radii_list.append(radius)

    # Convert lists to NumPy arrays
    if detected_centers_list:
        centers_array = np.array(detected_centers_list, dtype=float)
        radii_array = np.array(detected_radii_list, dtype=float)
    else:
        centers_array = np.empty((0, 2), dtype=float)
        radii_array = np.empty((0,), dtype=float)

    return centers_array, radii_array