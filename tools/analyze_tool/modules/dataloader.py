#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
dataloader.py

Manages loading and streaming of data for the analysis tool, including images and model output metadata.
"""

import os
import json
import pickle
import cv2

class Dataloader:
    def __init__(self, config, cameras):
        """
        Initialize the Dataloader with a dataset path and configuration.

        Args:
            config (dict): Configuration dictionary loaded from YAML. 
                           Expected keys:
                               - "base_path": Path to the dataset directory (str)
                               - "scenarios": List of scenario names or "all" (str or list)
                               - "output_path": Path for output data (str)
            cameras (dict): Dictionary of camera configurations. The keys of this dictionary
                            represent camera names. E.g., {"front_camera": {}, "rear_camera": {}}.
        """
        self.base_path = config["base_path"]
        # If "scenarios" is not explicitly specified, default to "all"
        self.scenarios = config.get("scenarios", "all")
        self.output_path = config["output_path"]

        # Store the camera names in a list for iteration and easy access
        self.cameras = list(cameras.keys())

        # Collect all frames (scenario, frame_id) pairs in a sorted manner
        self.frames = self._get_sorted_frames()

    def _get_sorted_frames(self):
        """
        Gather and return all frame IDs from each scenario folder, 
        combining frames found in both camera folders and the 'meta' folder.

        Returns:
            list: A sorted list of tuples (scenario_name, frame_id) where:
                  - scenario_name is the name of the scenario directory
                  - frame_id is the string identifier for a particular frame
        """
        frame_ids = []
        scenarios_to_load = []

        # Determine which scenarios to load (all subdirectories or a user-specified subset)
        if self.scenarios == "all":
            # List all directories in base_path (each directory represents a scenario)
            scenarios_to_load = [
                d for d in os.listdir(self.base_path) 
                if os.path.isdir(os.path.join(self.base_path, d))
            ]
        else:
            scenarios_to_load = self.scenarios

        # For each scenario, gather frame IDs from each camera folder plus the meta folder
        for scenario in scenarios_to_load:
            scenario_path = os.path.join(self.base_path, scenario)
            scenario_frames = set()  # Using a set to avoid duplicates

            # Check camera folders and 'meta' folder for files
            for folder in self.cameras + ["meta"]:
                folder_path = os.path.join(scenario_path, folder)
                if os.path.exists(folder_path):
                    # Consider files ending with .png or .json (we remove the extension)
                    files = [
                        f.split(".")[0] for f in os.listdir(folder_path) 
                        if f.endswith(".png") or f.endswith(".json")
                    ]
                    scenario_frames.update(files)

            # Sort the collected frame IDs and pair each with the scenario name
            for frame in sorted(scenario_frames):
                frame_ids.append((scenario, frame))

        return frame_ids

    def get_next_frame(self):
        """
        Generator that yields one frame of data at a time from each scenario. 
        This includes both images for each camera and model output metadata.

        Yields:
            dict: A dictionary with the structure:
                  {
                      "scenario": <scenario_name>,
                      "image": {
                          <camera_name_1>: <image_array_or_None>,
                          <camera_name_2>: <image_array_or_None>,
                          ...
                      },
                      "model_output": {
                          <key_from_json_or_pkl>: <value>,
                          ...
                      }
                  }
                  - "scenario": Name of the scenario the frame belongs to
                  - "image": A sub-dictionary of camera images
                  - "model_output": A sub-dictionary of loaded metadata from JSON / pickle
        """
        for scenario, frame_id in self.frames:
            # Initialize dictionary for the current frame
            frame_data = {
                "scenario": scenario,
                "image": {},
                "model_output": {}
            }
            scenario_path = os.path.join(self.base_path, scenario)

            # Load Images from each camera folder
            for cam in self.cameras:
                img_path = os.path.join(scenario_path, cam, f"{frame_id}.png")
                if os.path.exists(img_path):
                    frame_data["image"][cam] = cv2.imread(img_path)
                else:
                    # If the image file is missing, store None
                    frame_data["image"][cam] = None

            # Load model output (JSON) from the 'meta' folder
            pid_meta_path = os.path.join(scenario_path, "meta", f"{frame_id}.json")
            if os.path.exists(pid_meta_path):
                with open(pid_meta_path, "r") as f:
                    pid_meta = json.load(f)
                    for key in pid_meta:
                        frame_data["model_output"][key] = pid_meta[key]

            # Load model output (pickle) for predictions, if available
            pred_meta_path = os.path.join(scenario_path, "meta", f"{frame_id}_pred.pkl")
            if os.path.exists(pred_meta_path):
                with open(pred_meta_path, "rb") as f:
                    pred_meta = pickle.load(f)
                    for key in pred_meta:
                        frame_data["model_output"][key] = pred_meta[key]

            # Yield the populated frame data as a generator
            yield frame_data
