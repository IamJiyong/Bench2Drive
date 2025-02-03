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


def draw_bounding_boxes(
        image,
        box_corners,
        matrices,
        labels=None,
        color=(255, 0, 0),
        thickness=2):
    """
    Draw 3D bounding boxes on the given image, transforming from 3D world coordinates.

    This function expects each bounding box to be represented by 8 corner points in 3D:
    (x0, y0, z0), (x0, y0, z1), ..., (x1, y1, z0).

    Args:
        image (numpy.ndarray): The image on which to draw.
        boxes (list): A list of 8x3 array-like elements, each representing the 8 corners
                      of a 3D bounding box in world coordinates.
        labels (list, optional): A list of labels for each bounding box. Defaults to None.
        color (tuple, optional): The color of the bounding box lines in BGR. Defaults to (255, 0, 0).
        thickness (int, optional): Line thickness for drawing. Defaults to 2.
        camera_matrix (numpy.ndarray, optional): The camera intrinsic matrix (3x3).
        extrinsic_matrix (numpy.ndarray, optional): The camera extrinsic transformation matrix (4x4).
            - The top-left 3x3 portion is the rotation matrix.
            - The rightmost 3x1 portion is the translation vector.

    Returns:
        numpy.ndarray: The image with bounding boxes drawn on it.
    """
    if box_corners is None or len(box_corners) == 0:
        return image

    # Predefined edges for connecting the 8 corners of the 3D box
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # bottom rectangle
        (4, 5), (5, 6), (6, 7), (7, 4),  # top rectangle
        (0, 4), (1, 5), (2, 6), (3, 7)   # vertical connectors
    ]

    for i, box_3d in enumerate(box_corners):
        box_3d = np.array(box_3d, dtype=np.float32)

        # Validate that the box has shape (8, 3)
        if box_3d.shape != (8, 3):
            print(f"Warning: Unrecognized bounding box shape: {box_3d.shape}")
            continue

        corners_2d, behind_mask = project_3d_to_2d(box_3d, matrices, return_behind_mask=True)

        corners_2d = np.array(corners_2d, dtype=np.int32)

        # Ensure projected coordinates are within reasonable bounds
        mask1 = ((corners_2d[:, 0] > -1e5) & (corners_2d[:, 0] < 1e5) &
                    (corners_2d[:, 1] > -1e5) & (corners_2d[:, 1] < 1e5))
        
        # Ensure the bounding box size in 2D is reasonable
        mask2 = (corners_2d.max(axis=0) - corners_2d.min(axis=0) < 2000).all()
        
        # Combine both masks
        mask = mask1 & mask2 & ~behind_mask
        
        # Skip drawing if the bounding box is fully masked out
        if not mask.any():
            continue
        
        # Draw lines between the projected corners
        for start_idx, end_idx in edges:
            p1 = tuple(corners_2d[start_idx])
            p2 = tuple(corners_2d[end_idx])
            cv2.line(image, p1, p2, color, thickness)

        # If labels are provided, put text near the first corner
        if labels is not None and i < len(labels):
            label = str(labels[i])
            label_pos = tuple(corners_2d[0])
            cv2.putText(
                image,
                label,
                (label_pos[0], label_pos[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA
            )

    return image


def project_3d_to_2d(points_3d, matrices, return_behind_mask=False):
    """
    Projects 3D world coordinates to 2D image coordinates.

    Args:
        points_3d (numpy.ndarray): Nx3 array of 3D points.
        matrices (dict): Dictionary containing 'intrinsic' and 'extrinsic' matrices.
        return_behind_mask (bool): Whether to return a mask for points behind the camera.

    Returns:
        numpy.ndarray: Nx2 array of projected 2D points.
    """
    # TODO: Implement using combined intrinsic and extrinsic matrices (lidar2img)
    intrinsic, extrinsic = matrices['intrinsic'], matrices['extrinsic']
    rvec, tvec = extrinsic[:3, :3], extrinsic[:3, 3]

    # Convert to homogeneous coordinates (Nx4)
    points_4d = np.hstack((points_3d, np.ones((points_3d.shape[0], 1))))

    # Apply extrinsic transformation
    transformed_points = np.dot(rvec, points_4d[:, :3].T).T + tvec

    # Filter out points behind the camera
    behind_mask = transformed_points[:, 2] <= 0
    if not return_behind_mask:
        transformed_points = transformed_points[~behind_mask]

    # Convert to homogeneous coordinates (Nx4)
    transformed_points = np.hstack((transformed_points, np.ones((transformed_points.shape[0], 1))))

    # Compute projection matrix
    projection_matrix = np.dot(intrinsic, np.eye(3, 4))

    # Apply projection
    image_points = np.dot(projection_matrix, transformed_points.T).T

    # Normalize to get pixel coordinates
    image_points[:, 0] /= image_points[:, 2]
    image_points[:, 1] /= image_points[:, 2]

    if return_behind_mask:
        return image_points[:, :2], behind_mask
    else:
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

def overlay_trajectory(
        image,
        trajectory,
        matrices,
        is_ego=True,
        thickness=4):
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

    trajectory, behind_mask = project_3d_to_2d(np.array(trajectory), matrices, return_behind_mask=True)
    # TODO: for debugging behind_mask
    if np.any(behind_mask):
        return image

    if len(trajectory) < 2:
        return image
    
    trajectory = monotonic_spline(trajectory, num_points=500)

    y_coords = np.array([pt[1] for pt in trajectory])
    min_y, max_y = y_coords.min(), min(y_coords.max(), img_h)
    
    for i in range(len(trajectory) - 1):

        pt1 = (int(trajectory[i][0]), int(trajectory[i][1]))
        pt2 = (int(trajectory[i + 1][0]), int(trajectory[i + 1][1]))

        intensity = int(255 * (trajectory[i][1] - min_y) / (max_y - min_y))
        if is_ego:
            color = (255, 255 - intensity, 0)  # Light blue with intensity variation
        else:
            color = (0, 255 - intensity, 255)
        cv2.line(image, pt1, pt2, color=color, thickness=thickness)
    
    return image

def set_line_thickness(cam_name, element):
    if cam_name == "bev":
        thickness = 2
    else:
        thickness = 4
    
    return thickness

