#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
camera_calibration.py

Utility functions for computing camera intrinsic and extrinsic matrices.
"""

import yaml
import numpy as np
from scipy.spatial.transform import Rotation as R
import cv2

def compute_extrinsic_matrix(sensor_pose, ego_pose=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]):
    """
    Compute the extrinsic matrix from sensor to ego vehicle (reference frame at [0,0,0,0,0,0]).
    
    Args:
        sensor_pose (list): [x, y, z, roll, pitch, yaw] of the sensor.
        ego_pose (list, optional): Fixed ego vehicle pose (default is [0,0,0,0,0,0]).

    Returns:
        numpy.ndarray: 4x4 transformation matrix.
    """
    sensor_translation = np.array(sensor_pose[:3])
    sensor_rotation = np.array(sensor_pose[3:])

    sensor_rot_matrix = R.from_euler('xyz', sensor_rotation, degrees=True).as_matrix()
    sensor_extrinsic = np.eye(4)
    sensor_extrinsic[:3, :3] = sensor_rot_matrix
    sensor_extrinsic[:3, 3] = sensor_translation

    adjust_matrix = np.array([
        [1, 0,  0,  0],
        [0, 0, -1,  0],
        [0, 1,  0,  0],
        [0, 0,  0,  1]
    ])

    return adjust_matrix @ np.linalg.inv(sensor_extrinsic)

def load_camera_config(camera_data, camera_name):
    """
    Load camera intrinsic and extrinsic parameters for a specific camera.
    
    Args:
        camera_data (dict): Camera configuration dictionary.
        camera_name (str): Camera name to load parameters for (e.g., "rgb_front" or "bev").
    
    Returns:
        tuple: (intrinsic_matrix, extrinsic_matrix)
    """
    if camera_name not in camera_data:
        raise ValueError(f"Camera '{camera_name}' not found in configuration.")

    cam_data = camera_data[camera_name]

    intrinsic_matrix = np.array([
        [cam_data['intrinsic_matrix']['fx'], 0, cam_data['intrinsic_matrix']['cx']],
        [0, cam_data['intrinsic_matrix']['fy'], cam_data['intrinsic_matrix']['cy']],
        [0, 0, 1]
    ], dtype=np.float64)

    # Load pose from camera data
    pose = cam_data['pose']
    sensor_pose = [pose['x'], pose['y'], pose['z'], pose['roll'], pose['pitch'], pose['yaw']]
    extrinsic_matrix = compute_extrinsic_matrix(sensor_pose)

    return intrinsic_matrix, extrinsic_matrix


# # Example Usage
# intrinsic_rgb, extrinsic_rgb = load_camera_config("/home/ysh/jiyong/b2d_carla/Bench2Drive/tools/analyze_tool/configs/camera_config.yaml", "rgb_front")
# intrinsic_bev, extrinsic_bev = load_camera_config("/home/ysh/jiyong/b2d_carla/Bench2Drive/tools/analyze_tool/configs/camera_config.yaml", "bev")

# print("RGB Front Camera Extrinsic Matrix:\n", extrinsic_rgb)
# print("BEV Camera Extrinsic Matrix:\n", extrinsic_bev)
