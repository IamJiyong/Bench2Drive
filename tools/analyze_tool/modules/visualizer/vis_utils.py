#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
vis_utils.py

Utility functions for visualization tasks, such as drawing bounding boxes, overlaying trajectories, etc.
"""

import cv2
import numpy as np
from calibration_utils import project_3d_to_2d
import os

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

def overlay_trajectory(image, trajectory, color=(0, 0, 255), thickness=2, camera_matrix=None, extrinsic_matrix=None):
    """
    Overlay a trajectory (sequence of points) on the given image, transforming from 3D world coordinates if necessary.

    Args:
        image (numpy.ndarray): The image on which to overlay the trajectory.
        trajectory (list): A list of (x, y) points or 3D world coordinates.
        color (tuple, optional): The color of the trajectory in BGR format. Defaults to (0, 0, 255).
        thickness (int, optional): The thickness of the trajectory line. Defaults to 2.
        camera_matrix (numpy.ndarray, optional): Camera intrinsic matrix.
        extrinsic_matrix (numpy.ndarray, optional): Camera extrinsic transformation matrix.
    """
    if trajectory is None or len(trajectory) < 2:
        return

    if camera_matrix is not None and extrinsic_matrix is not None:
        rvec = extrinsic_matrix[:3, :3]
        tvec = extrinsic_matrix[:3, 3]
        trajectory = project_3d_to_2d(np.array(trajectory), camera_matrix, rvec, tvec)

    for i in range(len(trajectory) - 1):
        pt1 = (int(trajectory[i][0]), int(trajectory[i][1]))
        pt2 = (int(trajectory[i+1][0]), int(trajectory[i+1][1]))
        cv2.line(image, pt1, pt2, color, thickness)

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
