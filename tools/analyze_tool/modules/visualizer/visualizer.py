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

    def visualize_output(self, image, model_output, ground_truth=None):
        """
        Visualize the model output alongside the ground truth on the given image.

        Args:
            image (numpy.ndarray): The input image on which to overlay information.
            model_output (Any): The model's output data (e.g., bounding boxes, trajectories, etc.).
            ground_truth (Any): The ground truth data for comparison.
        """
        if image is None:
            print("No image provided for visualization.")
            return

        # Example usage of utility functions (assuming model_output might contain bounding boxes, trajectories, etc.)
        # if "boxes" in model_output:
        #     vis_utils.draw_bounding_boxes(image, model_output["boxes"], model_output["labels"])
        #
        # if "trajectory" in model_output:
        #     vis_utils.overlay_trajectory(image, model_output["trajectory"])

        # Optionally, visualize ground truth (similar approach)
        # if ground_truth and "trajectory" in ground_truth:
        #     vis_utils.overlay_trajectory(image, ground_truth["trajectory"], color=(0,255,0))

        # This function might not return anything if it draws in-place.
        # But if you want to keep the original image unmodified, consider copying first.
        return image

    def visualize_error(self, image, model_output, ground_truth):
        """
        Visualize the difference (error) between the model output and ground truth on the given image.

        Args:
            image (numpy.ndarray): The input image on which to overlay error information.
            model_output (Any): The model's output data.
            ground_truth (Any): The ground truth data.
        """
        if image is None:
            print("No image provided for error visualization.")
            return

        # TODO: Implement logic to visualize errors, such as drawing lines between
        # predicted and ground truth trajectory points or highlighting mismatched bounding boxes.
        # For example:
        # error_map = compute_error_map(model_output, ground_truth)
        # image_with_error = draw_error(image, error_map)
        pass

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