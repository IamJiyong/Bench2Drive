#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
visualizer.py

Handles data visualization logic.
Uses helper functions from vis_utils for drawing bounding boxes, trajectories, etc.
"""

import os
import cv2
import numpy as np
from . import vis_utils
from .vis_utils import VideoGenerator, set_line_thickness
from .camera_calibration import load_camera_config

class Visualizer:
    def __init__(self, config, cameras_config):
        """
        Initialize the Visualizer with the given configuration and camera parameters.

        Args:
            config (dict): A dictionary containing configuration parameters.
            cameras (dict): A dictionary containing camera calibration parameters.
        """
        self._config = config
        self.vis_elements = config.elements
        self.cameras_config = cameras_config  # Store camera configurations
        self.camera_matrices = self._load_camera_matrices(self._config.cameras)


    def _load_camera_matrices(self, cam_names):
        """Load intrinsic and extrinsic matrices for all cameras."""
        camera_matrices = {}
        for cam_name in cam_names:
            lidar2img_list = self.cameras_config[cam_name].get('lidar2img_matrix', None)

            if lidar2img_list is not None and False: # TODO: Not implemented yet
                lidar2img_mat = np.array(lidar2img_list, dtype=np.float64)
                camera_matrices[cam_name]['lidar2img_matrix'] = lidar2img_mat
            else:
                assert 'intrinsic_matrix' in self.cameras_config[cam_name], f"Missing intrinsic matrix for camera {cam_name}."
                assert 'extrinsic_matrix' in self.cameras_config[cam_name], f"Missing extrinsic matrix for camera {cam_name}."

                intrinsic_matrix, extrinsic_matrix = load_camera_config(self.cameras_config, cam_name)
                camera_matrices[cam_name] = {"intrinsic": intrinsic_matrix, "extrinsic": extrinsic_matrix}

        return camera_matrices


    def visualize_output(self, images, model_output, ground_truth=None):
        """
        Visualize model outputs on multiple camera images using camera matrices.

        Args:
            images (dict): A dictionary containing camera images keyed by camera name.
                        Example: {"rgb_front": <image_array>, "bev": <image_array>, ...}
            model_output (dict): A dictionary containing model predictions such as
                                planned trajectories, predicted trajectories, bounding boxes, etc.
        """
        # Validate that both inputs are provided
        if images is None or model_output is None:
            print("Invalid inputs for visualization: 'images' or 'model_output' is None.")
            return None

        # Iterate through each camera image
        for cam_name, image in images.items():
            # Check if the image is valid and whether we have camera matrices for this camera
            if image is None or cam_name not in self.camera_matrices:
                continue
            
            # Retrieve intrinsic and extrinsic matrices for the current camera
            matrices = self.camera_matrices[cam_name]
            
            # Overlay different visualization elements based on the specified vis_elements
            for element in self.vis_elements:
                thickness = set_line_thickness(cam_name, element)

                if 'trajectory' in element:
                    if element == "predicted_trajectory":
                        # Load the predicted trajectory from the model output
                        waypoints_2d = np.array(model_output["trajectories"], dtype=np.float32)
                        waypoints_2d = waypoints_2d.reshape(-1,waypoints_2d.shape[-2], 2)

                    elif element == "planned_trajectory":
                        # Load the planned trajectory from the model output
                        waypoints_2d = np.array(model_output["plan"], dtype=np.float32)
                        waypoints_2d = waypoints_2d.reshape(1,-1,2)

                    num_traj = waypoints_2d.shape[0]
                    for i in range(num_traj):
                        # TODO: hard-coded z-value (-1.6) for now
                        waypoints_3d = np.hstack((waypoints_2d[i], np.ones((waypoints_2d[i].shape[0], 1), dtype=np.float32) * (-1.6)))
                        image = vis_utils.overlay_trajectory(
                            image=image,
                            trajectory=waypoints_3d,
                            matrices=matrices,
                            thickness=thickness,
                            is_ego=element=="planned_trajectory",
                            )

                elif element == "bbox_3d":
                    # Draw bounding boxes
                    boxes_corners = model_output.get("boxes_corners", [])
                    image = vis_utils.draw_bounding_boxes(
                        image=image,
                        box_corners=boxes_corners,
                        matrices=matrices,
                        thickness=thickness,
                    )
            
            # Update the processed image in the dictionary
            images[cam_name] = image

        return images


    def save_visualization(self, images, image_metas, output_dir):
        """
        Save the visualization result (images) to the specified directory.

        Args:
            images (list): List of dictionaries of images to be saved.
                example: [{"rgb_front": <image_array>, "bev": <image_array>, ...}, ...]
            image_metas (dict): Dictionary of image metadata.
                'scenario': Name of the scenario
                'frame_id': Frame identifier
            output_dir (str): Directory where images will be saved.
        """
        # Create the output directory if it does not exist
        os.makedirs(output_dir, exist_ok=True)

        # # Save each image to the output directory
        for i, image in enumerate(images):
            scenario = image_metas[i]["scenario"]
            frame_id = image_metas[i]["frame_id"]

            for cam_name, cam_image in image.items():
                # Create a subdirectory for the scenario
                sub_dir = os.path.join(output_dir, scenario, cam_name)
                os.makedirs(sub_dir, exist_ok=True)

                img_path = os.path.join(sub_dir, f"{cam_name}_{frame_id}.png")
                cv2.imwrite(img_path, cam_image)

    def generate_video(self, images, image_metas):
        """
        Generate a video from a list of images.

        Args:
            images (list): A list of image frames (numpy arrays) to be converted to a video.
        """
        if not images:
            print("No images provided for video generation.")
            return

        # Initialize the video generator
        video_gen = VideoGenerator(self._config.video)

        # Generate the video
        video_gen.generate(images)
