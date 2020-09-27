import os
import numpy as np
from generate_instance import generate_instance, instance2json, save_json
from read_instance import load_instance

def test_generate_instance():
    ins = generate_instance(100)
    json_ins = instance2json(ins)
    save_json('ins-test.json', json_ins)
    ins_loaded = load_instance('ins-test.json')
    os.remove('ins-test.json')
    for a, b in zip(ins, ins_loaded):
        if isinstance(b, np.ndarray):
            assert np.array_equal(a, b)
        else:
            assert a == b
