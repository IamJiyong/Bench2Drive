import os
from dataloader import Dataloader
import yaml
from easydict import EasyDict

def load_yaml(file_path):
    """
    Recursively loads a YAML file and converts it into an EasyDict.
    If a value is a path to another YAML file, it loads that file as well.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"YAML file not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    
    def recursive_convert(d):
        if isinstance(d, dict):
            for key, value in d.items():
                if isinstance(value, str) and value.endswith('.yaml'):
                    d[key] = load_yaml(value)  # Recursively load nested YAML
                else:
                    d[key] = recursive_convert(value)
            return EasyDict(d)
        elif isinstance(d, list):
            return [recursive_convert(item) for item in d]
        else:
            return d
    
    return recursive_convert(data)

def test_dataloader():
    """ Test the Dataloader class."""
    # Load config
    config_path = "/workspace/Bench2Drive/tools/analyze_tool/configs/default.yaml"  # Change this to your actual config path
    config = load_yaml(config_path)
    
    # Initialize dataloader with config dictionary
    dataloader = Dataloader(config.dataloader_config)
    
    print("===== Testing Dataloader =====")
    print(f"Base Path: {dataloader.base_path}")
    print(f"Scenarios: {dataloader.scenarios}")
    print(f"Available Frames: {dataloader.frames[:10]}")  # Print first 10 frames

    if not dataloader.frames:
        print("⚠️ No frames found! Check if the base_path and scenarios are correct.")
        return

    # Test: Print first 3 frames
    num_test_frames = 3
    for i, frame_data in enumerate(dataloader.get_next_frame()):
        print(f"Scenario: {frame_data['scenario']}")
        print(f"Images: {list(frame_data['image'].keys())}")  # List available images
        print(f"Model Output: {'Available' if frame_data['model_output'] else 'Missing'}")

        if i >= num_test_frames - 1:
            break  # Stop after testing a few frames

    print("===== Test Completed =====")

if __name__ == "__main__":
    test_dataloader()
