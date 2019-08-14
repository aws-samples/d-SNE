"""
IO related functions
"""
import json
import pickle as pkl


def load_pkl(path):
    """
    Load pickle file
    :param path: file path
    :return:
    """
    with open(path, 'rb') as f:
        tr_x, tr_y, te_x, te_y = pkl.load(f)
        tr_y = tr_y.ravel().astype('int32')
        te_y = te_y.ravel().astype('int32')
    return tr_x, tr_y, te_x, te_y


def save_json(obj, path):
    """
    Save object to json file
    :param obj:
    :param path:
    :return:
    """
    with open(path, 'w') as f:
        json.dump(obj, f, indent=4, sort_keys=True)


def load_json(path):
    """
    Load json file
    :param path: file path
    :return:
    """
    with open(path) as f:
        cfg = json.load(f)
    return cfg
