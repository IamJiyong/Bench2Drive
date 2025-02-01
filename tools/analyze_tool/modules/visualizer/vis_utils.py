#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
vis_utils.py

Utility functions for visualization tasks, such as drawing bounding boxes, overlaying trajectories, etc.
"""

import cv2
import numpy as np
import os
from scipy.interpolate import PchipInterpolator

class VideoGenerator:
    def __init__(self, config):
        """
        Initialize the VideoGenerator with the given configuration.

        Args:
            config (dict): A dictionary containing configuration parameters.
        """
        self._config = config
        self.output_video = config.get("output_video", "output.mp4")

    def generate_video(self, results, output_path, fps=30):
        """
        Generates a video from a list of images (or a dictionary containing image paths/frames).

        Args:
            results (Any): The analysis results that contain or reference the images to be turned into a video.
            output_path (str): The file path where the generated video will be saved.
            fps (int): Frames per second for the output video.
        """
        # 1. Retrieve or generate a list of images (frames) from `results`.
        #    This could be direct image data, or file paths, or something else depending on your flow.
        image_list = self._extract_image_list(results)

        if not image_list:
            print("No images found to generate video.")
            return

        # 2. Create video writer
        #    For this, we need the width and height of the first image.
        first_frame = image_list[0]
        if isinstance(first_frame, str):
            # If it's a path, load the image
            first_frame = cv2.imread(first_frame)

        height, width, channels = first_frame.shape
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # or another codec
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # 3. Write frames to the video
        for i, frame_data in enumerate(image_list):
            if isinstance(frame_data, str):
                # If the frame is a file path
                frame = cv2.imread(frame_data)
            else:
                # If it's already an image (numpy array)
                frame = frame_data

            # (Optional) Add any overlay or additional info on the frame
            # frame = self.add_overlay(frame, info)

            video_writer.write(frame)

        video_writer.release()
        print(f"Video generation complete: {output_path}")

    def _extract_image_list(self, results):
        """
        Extract a list of images (or paths to images) from the analysis results.

        Args:
            results (Any): The analysis results that may contain image references.

        Returns:
            list: A list of image paths or image frames (numpy arrays).
        """
        # TODO: Implement logic to fetch/collect images for video generation.
        # For now, returning an empty list as a placeholder.
        return []

    def add_overlay(self, image, info):
        """
        Add additional information on the image/frame if needed.

        Args:
            image (numpy.ndarray): The frame to overlay info on.
            info (dict): A dictionary of info to overlay (e.g. text, shapes, etc.).

        Returns:
            numpy.ndarray: The modified image with overlays.
        """
        # TODO: Implement overlay logic, e.g., drawing bounding boxes, text, etc.
        return image


def draw_bounding_boxes(image, boxes, labels=None, color=(255, 0, 0), thickness=2, camera_matrix=None, extrinsic_matrix=None):
    """
    Draw bounding boxes on the given image, transforming from 3D world coordinates if necessary.

    Args:
        image (numpy.ndarray): The image on which to draw.
        boxes (list): A list of bounding box coordinates in either 2D (x_min, y_min, x_max, y_max) or 3D.
        labels (list, optional): A list of labels for each bounding box. Defaults to None.
        color (tuple, optional): The color of the bounding box in BGR format. Defaults to (255, 0, 0).
        thickness (int, optional): The thickness of the bounding box lines. Defaults to 2.
        camera_matrix (numpy.ndarray, optional): Camera intrinsic matrix.
        extrinsic_matrix (numpy.ndarray, optional): Camera extrinsic transformation matrix.
    """
    if boxes is None or len(boxes) == 0:
        return

    if camera_matrix is not None and extrinsic_matrix is not None:
        rvec = extrinsic_matrix[:3, :3]
        tvec = extrinsic_matrix[:3, 3]
        transformed_boxes = []
        for box in boxes:
            projected_box = project_3d_to_2d(np.array(box), camera_matrix, rvec, tvec)
            transformed_boxes.append(projected_box.flatten())
        boxes = transformed_boxes

    for i, box in enumerate(boxes):
        x_min, y_min, x_max, y_max = map(int, box)
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness)
        if labels is not None and i < len(labels):
            label = labels[i]
            cv2.putText(image, str(label), (x_min, y_min - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

def project_3d_to_2d(points_3d, intrinsic_matrix, rvec, tvec):
    """
    Projects 3D world coordinates to 2D image coordinates.

    Args:
        points_3d (numpy.ndarray): Nx3 array of 3D points.
        intrinsic_matrix (numpy.ndarray): 3x3 intrinsic camera matrix.
        rvec (numpy.ndarray): 3x3 rotation matrix.
        tvec (numpy.ndarray): 3x1 translation vector.

    Returns:
        numpy.ndarray: Nx2 array of projected 2D points.
    """
    if points_3d.shape[1] == 3:
        # Convert to homogeneous coordinates (Nx4)
        points_3d = np.hstack((points_3d, np.ones((points_3d.shape[0], 1))))

    # Apply extrinsic transformation
    transformed_points = np.dot(rvec, points_3d[:, :3].T).T + tvec

    # Filter out points behind the camera
    transformed_points = transformed_points[transformed_points[:, 2] > 0]

    # Convert to homogeneous coordinates (Nx4)
    transformed_points = np.hstack((transformed_points, np.ones((transformed_points.shape[0], 1))))

    # Compute projection matrix
    projection_matrix = np.dot(intrinsic_matrix, np.eye(3, 4))

    # Apply projection
    image_points = np.dot(projection_matrix, transformed_points.T).T

    # Normalize to get pixel coordinates
    image_points[:, 0] /= image_points[:, 2]
    image_points[:, 1] /= image_points[:, 2]

    return image_points[:, :2]


def monotonic_spline(points, num_points=100):
    """
    Generates a monotonic cubic spline curve through the given set of points.
    
    Args:
        points (list of tuples): List of control points [(x1, y1), (x2, y2), ...].
        num_points (int): Number of points to interpolate between control points.
    
    Returns:
        np.ndarray: Interpolated points along the spline.
    """
    if len(points) < 2:
        return np.array(points)

    points = np.array(points)
    points = points[np.argsort(points[:, 1])]

    x = points[:, 0]
    y = points[:, 1]

    pchip = PchipInterpolator(y, x)
    y_new = np.linspace(y.min(), y.max(), num_points)
    x_new = pchip(y_new)

    return np.column_stack((x_new, y_new))

def overlay_trajectory(cam_name, image, trajectory, intrinsic_matrix=None, extrinsic_matrix=None):
    """
    Overlay a trajectory (sequence of points) on the given image, transforming from 3D world coordinates if necessary.

    Args:
        image (numpy.ndarray): The image on which to overlay the trajectory.
        trajectory (list): A list of (x, y) points or 3D world coordinates.
        intrinsic_matrix (numpy.ndarray, optional): Camera intrinsic matrix.
        extrinsic_matrix (numpy.ndarray, optional): Camera extrinsic transformation matrix.
    """
    if trajectory is None or len(trajectory) < 2:
        return image
    
    img_h, img_w = image.shape[:2]  # Get image dimensions
    
    if intrinsic_matrix is not None and extrinsic_matrix is not None:
        rvec = extrinsic_matrix[:3, :3]
        tvec = extrinsic_matrix[:3, 3]
        trajectory = project_3d_to_2d(np.array(trajectory), intrinsic_matrix, rvec, tvec)

    # If camera is "bev", swap x and y coordinates
    if cam_name == "bev":
        trajectory = [(y, x) for x, y in trajectory]

    # Filter out points that are outside the image boundaries
    # trajectory = [(x, y) for x, y in trajectory if 0 <= x < img_w and 0 <= y < img_h]
    
    if len(trajectory) < 2:
        print("No valid trajectory points within image bounds.")
        return image
    
    trajectory = monotonic_spline(trajectory, num_points=100)

    y_coords = np.array([pt[1] for pt in trajectory])
    min_y, max_y = y_coords.min(), min(y_coords.max(), img_h)

    for i in range(len(trajectory) - 1):

        pt1 = (int(trajectory[i][0]), int(trajectory[i][1]))
        pt2 = (int(trajectory[i + 1][0]), int(trajectory[i + 1][1]))

        intensity = int(255 * (trajectory[i][1] - min_y) / (max_y - min_y))
        color = (255, 255 - intensity, 0)  # Light blue with intensity variation
        cv2.line(image, pt1, pt2, color=color, thickness=4)
    
    return image



def colorize_segmentation(mask):
    """
    Convert a segmentation mask to a colorized representation.

    Args:
        mask (numpy.ndarray): The segmentation mask, where each pixel 
                              may represent a class ID or a boolean value.

    Returns:
        numpy.ndarray: A colorized segmentation image (3-channel BGR).
    """
    if mask is None:
        return None

    unique_vals = np.unique(mask)
    color_map = {val: np.random.randint(0, 255, (3,)).tolist() for val in unique_vals}
    
    colorized = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for val in unique_vals:
        colorized[mask == val] = color_map[val]

    return colorized



def colorize_segmentation(mask):
    """
    Convert a segmentation mask to a colorized representation.

    Args:
        mask (numpy.ndarray): The segmentation mask, where each pixel 
                              may represent a class ID or a boolean value.

    Returns:
        numpy.ndarray: A colorized segmentation image (3-channel BGR).
    """
    if mask is None:
        return None

    # TODO: Implement the logic to map class IDs to specific colors
    # Example with random colors for demonstration
    unique_vals = np.unique(mask)
    color_map = {}
    for val in unique_vals:
        color_map[val] = np.random.randint(0, 255, (3,)).tolist()

    # Create an empty 3-channel image
    colorized = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

    for val in unique_vals:
        colorized[mask == val] = color_map[val]

    return colorized
