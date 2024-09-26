import copy
import numpy as np

def mean_dict(dictionaries, weights=None):
    if weights is None:
        weights = np.ones(len(dictionaries))

    result = {} 
    #copy.deepcopy(dictionaries[0])
    for key in dictionaries[0].keys():
        result[key] = np.average([dic[key] for dic in dictionaries], weights=weights, axis=0)
    return result