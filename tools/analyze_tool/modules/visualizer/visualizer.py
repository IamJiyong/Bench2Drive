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
from .vis_utils import VideoGenerator
from .camera_calibration import load_camera_config

class Visualizer:
    def __init__(self, config, cameras):
        """
        Initialize the Visualizer with the given configuration and camera parameters.

        Args:
            config (dict): A dictionary containing configuration parameters.
            cameras (dict): A dictionary containing camera calibration parameters.
        """
        self._config = config
        self.vis_elements = config.elements
        self.cameras = cameras  # Store camera configurations
        self.camera_matrices = self._load_camera_matrices()

    def _load_camera_matrices(self):
        """Load intrinsic and extrinsic matrices for all cameras."""
        camera_matrices = {}
        for cam_name in self.cameras:
            intrinsic, extrinsic = load_camera_config(self.cameras, cam_name)
            camera_matrices[cam_name] = {
                "intrinsic": intrinsic,
                "extrinsic": extrinsic
            }
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
            intrinsic_matrix = self.camera_matrices[cam_name]["intrinsic"]
            extrinsic_matrix = self.camera_matrices[cam_name]["extrinsic"]
            
            # Overlay different visualization elements based on the specified vis_elements
            for element in self.vis_elements:
                if element == "planned_trajectory":
                    # Overlay the planned trajectory
                    image = vis_utils.overlay_trajectory(
                        cam_name,
                        image,
                        model_output.get("plan", []),
                        intrinsic_matrix,
                        extrinsic_matrix,
                    )
                elif element == "predicted_trajectory":
                    # Overlay the predicted trajectory
                    image = vis_utils.overlay_trajectory(
                        cam_name,
                        image,
                        model_output.get("trajectory", []),
                        intrinsic_matrix,
                        extrinsic_matrix
                    )
                elif element == "boxes":
                    # Draw bounding boxes
                    image = vis_utils.draw_bounding_boxes(
                        image,
                        model_output.get("boxes", []),
                        intrinsic_matrix,
                        extrinsic_matrix
                    )
            
            # Update the processed image in the dictionary
            images[cam_name] = image

        return images


    def save_visualization(self, images, output_dir):
        """
        Save the visualization result (images) to the specified directory.

        Args:
            images (dict): Dictionary of images to be saved.
            output_dir (str): Directory where images will be saved.
        """
        if images is None:
            print("No images to save.")
            return

        os.makedirs(output_dir, exist_ok=True)
        for cam_name, image in images.items():
            if image is None:
                continue
            output_path = os.path.join(output_dir, f"{cam_name}.png")
            cv2.imwrite(output_path, image)
            print(f"Visualization saved at: {output_path}")

    def generate_video(self, images):
        """
        Generate a video from a list of images.

        Args:
            images (list): A list of image frames (numpy arrays) to be converted to a video.
        """
        if not images:
            print("No images provided for video generation.")
            return

        # Initialize the video generator
        video_gen = VideoGenerator(self._config.video.fps, self._config.video.output_path)

        # Generate the video
        video_gen.generate(images)
