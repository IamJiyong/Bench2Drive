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

class Visualizer:
    def __init__(self, config):
        """
        Initialize the Visualizer with the given configuration.

        Args:
            config (dict): A dictionary containing configuration parameters.
        """
        self._config = config
        self.vis_elements = config.elements

    def visualize_output(self, images, model_output, ground_truth=None):
        """
        Visualize the model output alongside the ground truth on the given image.

        Args:
            image (numpy.ndarray): The input image on which to overlay information.
            model_output (Any): The model's output data (e.g., bounding boxes, trajectories, etc.).
            ground_truth (Any): The ground truth data for comparison.
        """
        if images is None:
            print("No image provided for visualization.")
            return

        # Overlay visualization elements on the image
        for element in self.vis_elements:
            # Visualize the planned trajectory
            if element == "planned_trajectory":
                images = vis_utils.overlay_trajectory(images, model_output["planned_trajectory"])
            
            # Visualize the predicted trajectory
            elif element == "predicted_trajectory":
                images = vis_utils.overlay_trajectory(images, model_output["trajectory"])

            # Visualize bounding boxes
            elif element == "boxes":
                if ground_truth is not None and "boxes" in ground_truth:
                    gt_boxes = ground_truth["boxes"]
                else:
                    gt_boxes = []
                images = vis_utils.draw_bounding_boxes(images, model_output["boxes"], gt_boxes)
            
            # Add more visualization elements here
            else:
                pass

        return images


    def save_visualization(self, image, path):
        """
        Save the visualization result (image) to the specified path.

        Args:
            image (numpy.ndarray): The image to be saved.
            path (str): The file path where the image will be saved.
        """
        if image is None:
            print("No image to save.")
            return

        os.makedirs(os.path.dirname(path), exist_ok=True)
        cv2.imwrite(path, image)
        print(f"Visualization saved at: {path}")

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