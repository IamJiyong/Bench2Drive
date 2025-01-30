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
    def __init__(self, config):
        """
        Initialize the Dataloader with dataset path and configuration.

        Args:
            base_path (str): The base directory containing camera folders and metadata.
            config_path (str): Path to the YAML configuration file.
        """
        self.config = config
        self.cameras = self.config.cameras
        self.frames = self._get_sorted_frames()

    def _get_sorted_frames(self):
        """
        Get sorted frame numbers from all data folders.

        Returns:
            list: Sorted list of frame numbers as strings (e.g., ["0001", "0002", ...]).
        """
        frame_ids = set()
        for folder in self.cameras + ["model_output"]:
            folder_path = os.path.join(self.base_path, folder)
            if os.path.exists(folder_path):
                files = [f.split(".")[0] for f in os.listdir(folder_path) if f.endswith(".png") or f.endswith(".json")]
                frame_ids.update(files)

        return sorted(frame_ids)

    def get_next_frame(self):
        """
        Generator that yields one frame of data at a time.

        Yields:
            dict: A dictionary containing images (nested under "image") and model output metadata for a single frame.
        """
        for frame_id in self.frames:
            frame_data = {"image": {}, "model_output": None}

            # Load Images
            for cam in self.cameras:
                img_path = os.path.join(self.base_path, cam, f"{frame_id}.png")
                if os.path.exists(img_path):
                    frame_data["image"][cam] = cv2.imread(img_path)
                else:
                    frame_data["image"][cam] = None  # Handle missing images

            # Load Model Output (previously meta)
            model_output_path = os.path.join(self.base_path, "model_output", f"{frame_id}.json")
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
