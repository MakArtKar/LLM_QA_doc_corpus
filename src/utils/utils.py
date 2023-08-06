import os
import json


def get_config(name: str):
    path = os.path.join('../configs', name)
    with open(path) as f:
        config = json.load(f)
    return config
