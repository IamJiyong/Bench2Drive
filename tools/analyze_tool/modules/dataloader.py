#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
dataloader.py

Manages loading and streaming of data for the analysis tool, including images and model output metadata.
"""

import os
import json
import yaml
import cv2

class Dataloader:
    def __init__(self, config_path):
        """
        Initialize the Dataloader with dataset path and configuration.

        Args:
            config_path (str): Path to the YAML configuration file.
        """
        self.config = self._load_config(config_path)
        self.base_path = self.config["dataloader_config"]["base_path"]
        self.scenarios = self.config["dataloader_config"].get("scenarios", "all")
        self.output_path = self.config["dataloader_config"]["output_path"]
        self.cameras = self.config["dataloader_config"].get("cameras", [])
        self.frames = self._get_sorted_frames()

    def _load_config(self, config_path):
        """Load the configuration from a YAML file."""
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def _get_sorted_frames(self):
        """
        Get sorted frame numbers from all scenario data folders.

        Returns:
            list: Sorted list of (scenario, frame_id) tuples.
        """
        frame_ids = []
        scenarios_to_load = []

        if self.scenarios == "all":
            scenarios_to_load = [d for d in os.listdir(self.base_path) if os.path.isdir(os.path.join(self.base_path, d))]
        else:
            scenarios_to_load = self.scenarios

        for scenario in scenarios_to_load:
            scenario_path = os.path.join(self.base_path, scenario)
            scenario_frames = set()
            for folder in self.cameras + ["model_output"]:
                folder_path = os.path.join(scenario_path, folder)
                if os.path.exists(folder_path):
                    files = [f.split(".")[0] for f in os.listdir(folder_path) if f.endswith(".png") or f.endswith(".json")]
                    scenario_frames.update(files)
            for frame in sorted(scenario_frames):
                frame_ids.append((scenario, frame))

        return frame_ids

    def get_next_frame(self):
        """
        Generator that yields one frame of data at a time from each scenario.

        Yields:
            dict: A dictionary containing images (nested under "image") and model output metadata for a single frame.
        """
        for scenario, frame_id in self.frames:
            frame_data = {"scenario": scenario, "image": {}, "model_output": None}
            scenario_path = os.path.join(self.base_path, scenario)

            # Load Images
            for cam in self.cameras:
                img_path = os.path.join(scenario_path, cam, f"{frame_id}.png")
                if os.path.exists(img_path):
                    frame_data["image"][cam] = cv2.imread(img_path)
                else:
                    frame_data["image"][cam] = None  # Handle missing images

            # Load Model Output
            model_output_path = os.path.join(scenario_path, "meta", f"{frame_id}.json")
            if os.path.exists(model_output_path):
                with open(model_output_path, "r") as f:
                    frame_data["model_output"] = json.load(f)
                
            yield frame_data  # Generator 방식으로 반환

    def preprocess(self, frame_data):
        """
        Apply preprocessing to the images.

        Args:
            frame_data (dict): Dictionary containing images and model output metadata.

        Returns:
            dict: Preprocessed images and metadata.
        """
        for cam in self.cameras:
            if cam in frame_data["image"] and frame_data["image"][cam] is not None:
                frame_data["image"][cam] = cv2.cvtColor(frame_data["image"][cam], cv2.COLOR_BGR2RGB)
        
        return frame_data
