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

    def visualize_output(self, frame_data):
        """
        Visualize model output on a single frame's images using camera matrices.

        Args:
            frame_data (dict): Dictionary containing "image" (with individual camera images) and "model_output".
        """
        if "image" not in frame_data or "model_output" not in frame_data:
            print("Invalid frame data provided for visualization.")
            return None
        
        images = frame_data["image"]
        model_output = frame_data["model_output"]

        for cam_name, image in images.items():
            if image is None or cam_name not in self.camera_matrices:
                continue
            
            # Retrieve intrinsic and extrinsic matrices for the current camera
            intrinsic_matrix = self.camera_matrices[cam_name]["intrinsic"]
            extrinsic_matrix = self.camera_matrices[cam_name]["extrinsic"]
            
            # Overlay visualization elements on the image using camera matrices
            for element in self.vis_elements:
                if element == "planned_trajectory":
                    # vis_utils.py만 돌려보고 아직 여기서는 실행안해봄
                    image = vis_utils.overlay_trajectory(image, model_output.get("plan", []), intrinsic_matrix, extrinsic_matrix)
                elif element == "predicted_trajectory":
                    image = vis_utils.overlay_trajectory(image, model_output.get("trajectory", []), intrinsic_matrix, extrinsic_matrix)
                elif element == "boxes":
                    image = vis_utils.draw_bounding_boxes(image, model_output.get("boxes", []), intrinsic_matrix, extrinsic_matrix)
                
            images[cam_name] = image  # Update the image with overlays
        
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
