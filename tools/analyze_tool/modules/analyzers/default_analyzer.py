#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
base_analyzer.py

Provides a base class for all analyzers in the analysis tool. 
Child classes should override necessary methods to implement specialized analysis logic.
"""
from easydict import EasyDict

from ..dataloader import Dataloader
from ..visualizer.visualizer import Visualizer


class DefaultAnalyzer:
    def __init__(self, config:EasyDict, dataloader:Dataloader, visualizer:Visualizer):
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
        if self._config.visualize:
            assert self._visualizer is not None, "Visualizer is not initialized."
            results['visualize'] = self.visualize()
        
        # analyze model outputs. not implemented in this base class
        results['analysis'] = None

        return results

    def save_results(self, results, output_path):
        """
        Save the analysis results to the specified output path.

        Args:
            results (Any): The analysis results to be saved.
            output_path (str): The path to save the results.
        """
        if results.get('visualize', None) is not None:
            self._visualizer.save_visualization(images=results['visualize'][0],
                                                image_metas=results['visualize'][1],
                                                output_dir=output_path)
        if results.get('analysis', None) is not None:
            self._dataloader.save_data(results['analysis'], output_path)

    def visualize(self):
        """
        Visualize the given data using the visualizer, if visualization is enabled.

        Args:
            data (Any): The data to visualize (e.g., images, model outputs, etc.).
        """
        output = None

        # Visualize the data using the visualizer
        vis_images = []
        image_metas = []
        for data in self._dataloader.get_next_frame():
            image_meta = {
                "scenario": data["scenario"],
                "frame_id": data["frame_id"],
            }

            images = data["image"]
            model_output = data["model_output"]
            ground_truth = data.get('ground_truth', None)
            vis_image = self._visualizer.visualize_output(images, model_output, ground_truth)

            image_metas.append(image_meta)
            vis_images.append(vis_image)
        
        # Generate video or return list of images
        # TODO: seperate videos per scenario if specified
        if self._visualizer._config.output_video:
            video = self._visualizer.generate_video(
                vis_images,
                image_metas,
                )
            output = (video, None)
        if self._visualizer._config.output_images:
            output = (vis_images, image_metas)

        return output
