import os
import csv
import argparse
from collections import OrderedDict

def init_config(config, default_config, name=None):
    """Initialise non-given config values with defaults"""
    if config is None:
        config = default_config
    else:
        for k in default_config.keys():
            if k not in config.keys():
                config[k] = default_config[k]
    # Ensure PRINT_CONFIG is handled globally
    if 'PRINT_CONFIG' not in config:
        config['PRINT_CONFIG'] = False
    if name and config.get('PRINT_CONFIG', False):  # Use get to avoid KeyError
        print('\n%s Config:' % name)
        for c in config.keys():
            print('%-20s : %-30s' % (c, config[c]))
    return config

def update_config(config):
    """
    Parse the arguments of a script and updates the config values for a given value if specified in the arguments.
    :param config: the config to update
    :return: the updated config
    """
    parser = argparse.ArgumentParser()
    for setting in config.keys():
        if setting in ['GT_PATH', 'TRACKER_PATH']:
            parser.add_argument(f"--{setting}", type=str)
        elif isinstance(config[setting], list) or config[setting] is None:
            parser.add_argument(f"--{setting}", nargs='+')
        else:
            parser.add_argument(f"--{setting}")
    args = parser.parse_args().__dict__
    for setting in args.keys():
        if args[setting] is not None:
            if isinstance(config[setting], bool):
                if args[setting] == 'True':
                    x = True
                elif args[setting] == 'False':
                    x = False
                else:
                    raise Exception(f'Command line parameter {setting} must be True or False')
            elif isinstance(config[setting], int):
                x = int(args[setting])
            elif args[setting] is None:
                x = None
            else:
                x = args[setting]
            config[setting] = x
    return config

def get_code_path():
    """Get base path where code is"""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def validate_metrics_list(metrics_list):
    """Get names of metric class and ensures they are unique, further checks that the fields within each metric class
    do not have overlapping names.
    """
    metric_names = [metric.get_name() for metric in metrics_list]
    # check metric names are unique
    if len(metric_names) != len(set(metric_names)):
        raise TrackEvalException('Code being run with multiple metrics of the same name')
    fields = []
    for m in metrics_list:
        fields += m.fields
    # check metric fields are unique
    if len(fields) != len(set(fields)):
        raise TrackEvalException('Code being run with multiple metrics with fields of the same name')
    return metric_names

class TrackEvalException(Exception):
    """Custom exception for catching expected errors."""
    ...