#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
analyze.py

Entry point for the analysis tool. 
Loads the configuration, sets up the analyzer, and runs the analysis.

"""

import argparse
import yaml
from easydict import EasyDict as edict

# Example imports (adjust or remove if you do not need all)
from modules.analyzers import __all__ as analyzer_dict

def main():
    """Main function to run the analysis tool."""

    # 1. Parse command line arguments (optional)
    parser = argparse.ArgumentParser(description="Run the analysis tool.")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to the configuration file.")
    args = parser.parse_args()

    # 2. Load configuration from YAML file and convert to EasyDict
    with open(args.config, "r") as f:
        config = edict(yaml.safe_load(f))

    # 3. Choose which analyzer to use based on config
    analyze_type = config.analyze.type if "analyze" in config and "type" in config.analyze else "base"
    analyzer = analyzer_dict[analyze_type](config)

    # 4. Run the analysis
    results = analyzer_dict[analyze_type].analyze()

    # 5. Save or output the results
    analyzer.save_results(results, config.data.output_path)

if __name__ == "__main__":
    main()
