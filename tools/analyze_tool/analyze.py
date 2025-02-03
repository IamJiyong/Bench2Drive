#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
analyze.py

Entry point for the analysis tool. 
Loads the configuration, sets up the analyzer, and runs the analysis.

"""

import argparse

from modules.dataloader import Dataloader
from modules.visualizer.visualizer import Visualizer
from modules.analyzers import __all__ as analyzer_dict
from utils.config import load_yaml


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run the analysis tool.")
    parser.add_argument("--config", type=str, default="tools/analyze_tool/configs/default.yaml",
                        help="Path to the configuration file.")
    args = parser.parse_args()
    return args


def main():
    """Main function to run the analysis tool."""

    # 1. Parse command line arguments (optional)
    args = parse_arguments()

    # 2. Load configuration from YAML file and convert to EasyDict
    config = load_yaml(args.config)

    # 3. Initialize dataloader and visualizer
    # TODO: dataloader_config만 넘기기
    dataloader = Dataloader(config.dataloader_config, config.visualizer_config)
    if config.get("visualizer_config", None) is not None:
        visualizer = Visualizer(config.visualizer_config, config.cameras, config.output_path)
    else:
        visualizer = None

    # 4. Choose which analyzer to use based on config
    analyzer = analyzer_dict[config.analyze_config.type](config.analyze_config, dataloader, visualizer)

    # 5. Run the analysis
    results = analyzer.analyze()

if __name__ == "__main__":
    main()
