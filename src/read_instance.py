import json
import numpy as np
from glob import glob


def list_avaliable_instances(regex="data/*.json"):
    return glob(regex)


def read_json(filename):
    with open(filename, "r") as file:
        data = json.loads(file.read())
    return data


def load_instance(filename):
    json_ins = read_json(filename)
    return [
        json_ins["j"],
        json_ins["n_F"],
        json_ins["n_L"],
        json_ins["n_M"],
        json_ins["n_P"],
        np.array(json_ins["D"]),
        np.array(json_ins["r"]),
        np.array(json_ins["R"]),
        np.array(json_ins["C"]),
        np.array(json_ins["p"]),
        np.array(json_ins["t"]),
    ]
