import yaml
import os
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
