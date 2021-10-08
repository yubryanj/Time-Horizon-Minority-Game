import json
import argparse

def get_configs(
    path = './configs.json',
    ):
    """
    Retrieves the environment configuration
    :args   path    path to the config file
    :output data    configs in json format
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        type=str,   
        default='default'
    )
    args = parser.parse_args()

    with open(path) as f:
        data = f.read()
    data = json.loads(data)

    configs = data[args.experiment]

    return configs
