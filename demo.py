import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import multiprocessing

import sobel_edge
import log_edge
import circ_hough

def worker_hough_computation(v_min_arg, hough_input_edges_01_arg, R_max_hough_arg, dim_hough_arg):
    print(f"Worker started for V_min: {v_min_arg}")
    start_time_hough = time.time()
    centers, radii = circ_hough.circ_hough(hough_input_edges_01_arg, R_max_hough_arg, dim_hough_arg, v_min_arg)
    duration_hough = time.time() - start_time_hough
    print(f"Worker for V_min: {v_min_arg} finished in {duration_hough:.2f} seconds, found {len(radii)} circles.")
    return v_min_arg, centers, radii, duration_hough

def display_image(title, image, cmap=None):
    """Helper function to display images using matplotlib"""
    plt.figure(figsize=(8, 6))
    plt.imshow(image, cmap=cmap)
    plt.title(title)
    plt.axis('off')
    plt.show()

def display_images_side_by_side(title1, image1, title2, image2, cmap1=None, cmap2=None):
    """Helper function to display two images side by side"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(image1, cmap=cmap1)
    axes[0].set_title(title1)
    axes[0].axis('off')
    axes[1].imshow(image2, cmap=cmap2)
    axes[1].set_title(title2)
    axes[1].axis('off')
    plt.tight_layout()
    plt.show()

def main():
    output_dir = "output_images"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # --- 1. Load and Prepare Image ---
    img_path = 'basketball_large.png'
    if not os.path.exists(img_path):
        print(f"Error: Image file not found at {img_path}")
        print("Please ensure 'basketball_large.png' is in the same directory as demo.py.")
        return

    original_img_color = cv2.imread(img_path)
    if original_img_color is None:
        print(f"Error: Could not load image from {img_path}. Check the file format and path.")
        return
        
    original_img_color = cv2.cvtColor(original_img_color, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(original_img_color, cv2.COLOR_RGB2GRAY)
    
    # Normalize grayscale image to [0, 1] for Sobel and LoG
    img_gray_normalized = img_gray.astype(float) / 255.0

    print("--- Original Grayscale Image ---")
    fig_gray, ax_gray = plt.subplots(figsize=(10, 8))
    ax_gray.imshow(img_gray_normalized, cmap='gray')
    ax_gray.set_title('Original Grayscale Image')
    ax_gray.axis('off')
    plt.savefig(os.path.join(output_dir, "original_grayscale.png"))
    plt.close(fig_gray)

    # --- 2. Sobel Edge Detection Tests ---
    print("\n--- Sobel Edge Detection ---")
    sobel_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
    detected_edge_counts = []

    for thres in sobel_thresholds:
        print(f"Applying Sobel with threshold: {thres}")
        sobel_output_thres = sobel_edge.sobel_edge(img_gray_normalized, thres)
        # Convert to 0-1 range if it's 0-255
        if np.max(sobel_output_thres) > 1:
            sobel_output_thres = sobel_output_thres / 255.0
        
        fig_sobel_comp, axes_sobel_comp = plt.subplots(1, 2, figsize=(16, 8))
        axes_sobel_comp[0].imshow(img_gray_normalized, cmap='gray')
        axes_sobel_comp[0].set_title('Original Grayscale')
        axes_sobel_comp[0].axis('off')
        axes_sobel_comp[1].imshow(sobel_output_thres, cmap='gray')
        axes_sobel_comp[1].set_title(f'Sobel Edges (Thres={thres})')
        axes_sobel_comp[1].axis('off')
        fig_sobel_comp.suptitle(f'Sobel Edge Detection vs Original (Threshold {thres})')
        plt.savefig(os.path.join(output_dir, f"sobel_comparison_thres{thres:.1f}.png"))
        plt.close(fig_sobel_comp)

        edge_count = np.sum(sobel_output_thres == 1)
        detected_edge_counts.append(edge_count)

    # Plot number of detected points vs. threshold
    plt.figure(figsize=(8, 5))
    plt.plot(sobel_thresholds, detected_edge_counts, marker='o')
    plt.title('Sobel: Number of Detected Edge Points vs. Threshold')
    plt.xlabel('Threshold Value')
    plt.ylabel('Number of Detected Edge Points')
    plt.grid(True)
    plt.show()

    # --- 3. LoG Edge Detection Test ---
    print("\n--- LoG Edge Detection ---")
    log_edges_01 = log_edge.log_edge(img_gray_normalized) # Returns 0 or 1
    log_edges_01_displayable = (log_edges_01 * 255).astype(np.uint8) # Convert to 0/255 for display/saving
    print(f"LoG detected {np.sum(log_edges_01)} edge points.")

    fig_log_comp, axes_log_comp = plt.subplots(1, 2, figsize=(16, 8))
    axes_log_comp[0].imshow(img_gray_normalized, cmap='gray')
    axes_log_comp[0].set_title('Original Grayscale')
    axes_log_comp[0].axis('off')
    axes_log_comp[1].imshow(log_edges_01_displayable, cmap='gray')
    axes_log_comp[1].set_title('LoG Edges')
    axes_log_comp[1].axis('off')
    fig_log_comp.suptitle('LoG Edge Detection vs Original')
    plt.savefig(os.path.join(output_dir, "log_comparison.png"))
    plt.close(fig_log_comp)


    # --- 4. Circular Hough Transform Tests (with Image Rescaling) ---
    print("\n--- Circular Hough Transform ---")
    chosen_sobel_thres_for_hough = 0.4
    print(f"Using Sobel edges (threshold={chosen_sobel_thres_for_hough}) for Hough input.")

    # Rescale image for Hough transform to improve speed
    hough_rescale_factor = 0.25 # Reverted rescale factor for speed
    print(f"Rescaling image by {hough_rescale_factor} for Hough input.")
    img_gray_normalized_scaled_h = cv2.resize(
        img_gray_normalized, 
        (0,0),
        fx=hough_rescale_factor, 
        fy=hough_rescale_factor, 
        interpolation=cv2.INTER_AREA
    )

    hough_input_edges_255 = sobel_edge.sobel_edge(img_gray_normalized_scaled_h, thres=chosen_sobel_thres_for_hough)
    
    # circ_hough expects a binary image with values 0 or 1
    hough_input_edges_01 = (hough_input_edges_255 / 255).astype(int)

    display_images_side_by_side(
        'Original Color', original_img_color,
        f'Input to Hough (Sobel on {hough_rescale_factor*100}% Rescaled Img)', hough_input_edges_01,
        cmap2='gray'
    )

    # Parameters for Hough Transform are now based on the SCALED image dimensions
    img_h_scaled, img_w_scaled = hough_input_edges_01.shape
    R_max_hough = min(img_h_scaled, img_w_scaled) * 0.6
    
    dim_hough = np.array([img_w_scaled // 8, img_h_scaled // 8, 40], dtype=int) 
    
    hough_V_min_values = [4000, 5000, 6000, 7000, 8000]

    print(f"Hough Parameters: R_max={R_max_hough:.2f}, Accumulator Dims={dim_hough}")

    # Prepare arguments for parallel processing
    pool_args = [(v_min, hough_input_edges_01, R_max_hough, dim_hough) for v_min in hough_V_min_values]

    # Use multiprocessing Pool
    num_processes = min(len(hough_V_min_values), os.cpu_count() if os.cpu_count() else 1)
    print(f"\nStarting Hough transform processing with {num_processes} parallel processes...")
    
    results = []
    if __name__ == '__main__':
        with multiprocessing.Pool(processes=num_processes) as pool:
            results = pool.starmap(worker_hough_computation, pool_args)

    print("\nAll Hough computations finished. Processing results sequentially...")
    results.sort(key=lambda x: x[0])

    for v_min_res, centers_res, radii_res, duration_res in results:
        print(f"\nResults for V_min: {v_min_res}")
        print(f"Hough transform took {duration_res:.2f} seconds.")
        print(f"Detected {len(radii_res)} circles.")

        img_with_circles = original_img_color.copy()

        # Draw detected circles
        for i in range(len(radii_res)):
            a_scaled, b_scaled = centers_res[i]
            r_scaled = radii_res[i]
            a_orig = int(a_scaled / hough_rescale_factor)
            b_orig = int(b_scaled / hough_rescale_factor)
            r_orig = int(r_scaled / hough_rescale_factor)
            cv2.circle(img_with_circles, (a_orig, b_orig), r_orig, (0, 255, 0), 2)

        # Display the image with detected circles
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(cv2.cvtColor(img_with_circles, cv2.COLOR_BGR2RGB))
        ax.set_title(f'Detected Circles (V_min={v_min_res}, Sobel Thres={chosen_sobel_thres_for_hough}, Scale={hough_rescale_factor}) - {len(radii_res)} circles')
        ax.axis('off')
        plt.savefig(os.path.join(output_dir, f"hough_circles_vmin{v_min_res}_sobel{chosen_sobel_thres_for_hough}_scale{hough_rescale_factor}.png"))
        plt.close(fig) 

    print("\n--- Demo Finished ---")

if __name__ == '__main__':
    main()