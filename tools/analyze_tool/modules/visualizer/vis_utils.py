#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
vis_utils.py

Utility functions for visualization tasks, such as drawing bounding boxes, overlaying trajectories, etc.
"""

import cv2
import numpy as np

def draw_bounding_boxes(image, boxes, labels=None, color=(255, 0, 0), thickness=2):
    """
    Draw bounding boxes on the given image.

    Args:
        image (numpy.ndarray): The image on which to draw.
        boxes (list): A list of bounding box coordinates, 
                      where each box is (x_min, y_min, x_max, y_max).
        labels (list, optional): A list of labels for each bounding box. 
                                 Defaults to None.
        color (tuple, optional): The color of the bounding box in BGR format. 
                                 Defaults to (255, 0, 0).
        thickness (int, optional): The thickness of the bounding box lines. 
                                   Defaults to 2.
    """
    if boxes is None or len(boxes) == 0:
        return

    for i, box in enumerate(boxes):
        x_min, y_min, x_max, y_max = box
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness)
        if labels is not None and i < len(labels):
            label = labels[i]
            # Optionally, draw label text above the bounding box
            cv2.putText(image, str(label), (x_min, y_min - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def overlay_trajectory(image, trajectory, color=(0, 0, 255), thickness=2):
    """
    Overlay a trajectory (sequence of points) on the given image.

    Args:
        image (numpy.ndarray): The image on which to overlay the trajectory.
        trajectory (list): A list of (x, y) points.
        color (tuple, optional): The color of the trajectory in BGR format. 
                                 Defaults to (0, 0, 255).
        thickness (int, optional): The thickness of the trajectory line. 
                                   Defaults to 2.
    """
    if trajectory is None or len(trajectory) < 2:
        return

    for i in range(len(trajectory) - 1):
        pt1 = (int(trajectory[i][0]), int(trajectory[i][1]))
        pt2 = (int(trajectory[i+1][0]), int(trajectory[i+1][1]))
        cv2.line(image, pt1, pt2, color, thickness)

    # Optionally, draw points themselves
    # for (x, y) in trajectory:
    #     cv2.circle(image, (int(x), int(y)), 3, color, -1)


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
