#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
base_analyzer.py

Provides a base class for all analyzers in the analysis tool. 
Child classes should override necessary methods to implement specialized analysis logic.
"""

from ..dataloader import Dataloader
from ..visualizer.visualizer import Visualizer


class DefaultAnalyzer:
    def __init__(self, config, dataloader, visualizer):
        """
        Initialize the base analyzer with config.

        Args:
            config (dict): Dictionary containing configuration parameters.
        """
        self._dataloader = dataloader
        self._visualizer = visualizer
        self._config = config

    def analyze(self):
        """
        Perform the main analysis flow.
        Override this method in child classes to implement specific analysis logic.

        Returns:
            Any: The analysis results (could be metrics, processed data, etc.).
        """
        # Example: load data, do minimal processing, and return results.
        results = {}

        # visualize the data if visualizer is enabled
        if self._visualizer and self._config.visualize:
            results['visualize'] = self.visualize()

        return results

    def save_results(self, results, output_path):
        """
        Save the analysis results to the specified output path.

        Args:
            results (Any): The analysis results to be saved.
            output_path (str): The path to save the results.
        """
        # Example: use dataloader to save data
        self._dataloader.save_data(results, output_path)

    def visualize(self):
        """
        Visualize the given data using the visualizer, if visualization is enabled.

        Args:
            data (Any): The data to visualize (e.g., images, model outputs, etc.).
        """
        output = None

        # Visualize the data using the visualizer
        vis_images = []
        for data in self._dataloader.get_next_frame():
            images = data["image"]
            model_output = data["model_output"]
            ground_truth = data.get('ground_truth', None)
            vis_image = self._visualizer.visualize_output(images, model_output, ground_truth)
            vis_images.append(vis_image)
        
        # Generate video or return list of images
        if self._visualizer.output_video:
            video = self._visualizer.generate_video(vis_images, self._config["visualize"]["output_path"])
            output = video
        else:
            output = vis_images

        return output
                
