import cv2
import numpy as np
from vis_utils import overlay_trajectory
import matplotlib.pyplot as plt

def test_overlay_trajectory(image_path, trajectory, intrinsic_matrix, extrinsic_matrix, output_path):
    """
    Test function to visualize the trajectory overlay on an image.
    
    Args:
        image_path (str): Path to the test image.
        trajectory (list): List of (x, y, z) coordinates in 3D.
        intrinsic_matrix (numpy.ndarray): Camera intrinsic matrix.
        extrinsic_matrix (numpy.ndarray): Camera extrinsic transformation matrix.
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    # Overlay trajectory
    output_image = overlay_trajectory(image, trajectory, intrinsic_matrix=intrinsic_matrix, extrinsic_matrix=extrinsic_matrix)
    
    # Display result
    # Save the result using OpenCV
    cv2.imwrite(output_path, output_image)
    print(f"✅ Saved visualization as {output_path}")



if __name__ == "__main__":
    # Define test parameters
    test_image_path = "/workspace/Bench2Drive/eval_v1/RouteScenario_1711_rep0_Town12_ParkingCutIn_1_15_01_27_10_23_27/rgb_front/0000.png"  # Set actual path
    test_output_path = "/workspace/Bench2Drive/eval_v1/test_prjection.png"
    test_trajectory = [
        (0, 0.224, 0), (0, 1.028, 0), (0, 2.530, 0), (-0.136, 4.680, 0), (-0.125, 7.3770, 0), (-0.137, 10.390, 0)
    ]  # Example trajectory in 3D
    
    test_intrinsic_matrix = np.array([
        [1142.5184, 0, 800],
        [0, 1142.5184, 450],
        [0, 0, 1]
    ])
    
    test_extrinsic_matrix = np.array([
        [1, 0, 0, -0.8],
        [0, 0, -1, 1.6],
        [0, 1, 0, 0],
        [0, 0, 0, 1]
    ])
    
    test_overlay_trajectory(test_image_path, test_trajectory, test_intrinsic_matrix, test_extrinsic_matrix, test_output_path)
