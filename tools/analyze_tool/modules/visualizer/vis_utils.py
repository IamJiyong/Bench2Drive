#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
vis_utils.py

Utility functions for visualization tasks, such as drawing bounding boxes, overlaying trajectories, etc.
"""

import cv2
import numpy as np
from calibration_utils import project_3d_to_2d

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
