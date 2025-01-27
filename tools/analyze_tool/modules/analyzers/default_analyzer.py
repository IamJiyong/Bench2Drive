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
    def __init__(self, config):
        """
        Initialize the base analyzer with config.

        Args:
            config (dict): Dictionary containing configuration parameters.
        """
        self._dataloader = Dataloader(config.data_config) # initialize dataloader with data config
        self._visualizer = Visualizer(config.visualizer_config) # initialize visualizer with visualizer config
        self._config = config

    def analyze(self):
        """
        Perform the main analysis flow.
        Override this method in child classes to implement specific analysis logic.

        Returns:
            Any: The analysis results (could be metrics, processed data, etc.).
        """
        # Example: load data, do minimal processing, and return results.
        input_path = self._config["data"]["input_path"]
        data = self._dataloader.load_data(input_path)
        # No specific analysis here; just return the loaded data as a placeholder.
        return data

    def save_results(self, results, output_path):
        """
        Save the analysis results to the specified output path.

        Args:
            results (Any): The analysis results to be saved.
            output_path (str): The path to save the results.
        """
        # Example: use dataloader to save data
        self._dataloader.save_data(results, output_path)

    def visualize(self, data):
        """
        Visualize the given data using the visualizer, if visualization is enabled.

        Args:
            data (Any): The data to visualize (e.g., images, model outputs, etc.).
        """
        visualize_enabled = self._config.get("visualize", {}).get("enable", False)
        if visualize_enabled:
            # Example of calling a visualizer method
            # This method would need to be adapted to the actual data structure
            if isinstance(data, dict):
                image = data.get("image")
                model_output = data.get("model_output")
                ground_truth = data.get("ground_truth")
                self._visualizer.visualize_output(image, model_output, ground_truth)
