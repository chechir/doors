import json
import os
import pickle

import pandas as pd


def append_csv(data, path):
    """append to a csv"""
    assert path.endswith(".csv")
    to_log = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
    if "~" in path:
        print("Dont include the user path as ~!, use os.path.expanduser")
    if os.path.isfile(path):
        current_log = pd.read_csv(path)
        current_log = current_log.append(to_log, ignore_index=True, sort=True)
    else:
        current_log = to_log
    current_log.to_csv(path, index=False)


def ensure_dir_exists(path):
    """Takes path like:
       /path/to/test.pkl  OR
       /path/to/
    Ensures directory exists so file can be writen
    """
    if "." in path.split("/")[-1]:
        directory = "/".join(path.split("/")[:-1])
    else:
        directory = path
    if not os.path.exists(directory):
        os.makedirs(directory)


def read_pickle(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj


def write_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def write_json(obj, path, encoder=None):
    with open(path, "wb") as f:
        if encoder is None:
            json.dump(obj, f)
        else:
            json.dump(obj, f, cls=encoder)


def read_json(path):
    with open(path, "rb") as f:
        obj = json.load(f)
    return obj


def read_json_per_line(path):
    data = []
    with open(path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def write_json_per_line(list_of_data, path):
    """Each element in list_of_data will be written
    to path as a json on its own line."""
    ensure_dir_exists(path)
    assert isinstance(list_of_data, list), "data must be list"
    with open(path, "a") as fhandler:
        for row in list_of_data:
            json_formatted_data = json.dumps(row, cls=json.JSONEncoder)
            fhandler.write(json_formatted_data + "\n")
