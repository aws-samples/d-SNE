"""
Basic functions for the datasets loaders
"""
import os
import random
import warnings
import numpy as np

from .io import load_pkl


def list_images(root):
    """
    List the images in folder
    :param root: folder path
    :return: class, (image path and ID)
    """
    set_s = []
    items = []

    for folder in sorted(os.listdir(root)):
        path = os.path.join(root, folder)
        if not os.path.isdir(path):
            warnings.warn('Ignoring %s, which is not a directory.' % path, stacklevel=3)
            continue
        label = len(set_s)
        set_s.append(folder)
        for filename in sorted(os.listdir(path)):
            filename = os.path.join(path, filename)
            ext = os.path.splitext(filename)[1]
            if ext.lower() not in ['.jpg', '.jpeg', '.png']:
                warnings.warn('Ignoring %s of type %s. Only support %s' % (
                    filename, ext, ', '.join(['.jpg', '.jpeg', '.png'])))
                continue
            items.append((filename, label))
    return set_s, items


def sampling_office(items, sm, sns, snt, seed=0):
    """
    Sampling method for office dataset
    :param items: items (path, label)
    :param sm: sampling method
    :param sns: sampling number for source
    :param snt: sampling number for target
    :param seed: random seed
    :return:
        trs: training source set
        trt: training target set
        tet: testing target set
    """
    rng = random.Random(seed)

    # build dictionary
    cls = {}
    for idx, (p, l) in enumerate(items):
        if l in cls:
            cls[l].append(idx)
        else:
            cls[l] = [idx]

    # select source train, target train, target test samples
    trs, trt, tet = [], [], []
    if sm == 0:
        # random sampling

        rng.shuffle(items)
        for idx in range(0, snt):
            trt.append(items[idx])

        for idx in range(snt, len(items)):
            tet.append(items[idx])

        rng.shuffle(items)
        for idx in range(0, sns):
            trs.append(items[idx])
    elif sm == 1:
        # balanced sampling
        n_snt = snt // len(cls)
        n_sns = sns // len(cls)
        for k, v in cls.items():
            rng.shuffle(v)
            trt.extend(v[:n_snt])
            tet.extend(v[n_snt:])
            rng.shuffle(v)
            if n_sns == 0:
                trs.extend(v)
            else:
                trs.extend(v[:n_sns])

        trs = [items[idx] for idx in trs]
        trt = [items[idx] for idx in trt]
        tet = [items[idx] for idx in tet]
    else:
        n_snt = snt // len(cls)
        for k, v in cls.items():
            rng.shuffle(v)
            trt.extend(v[:n_snt])
            trs.extend(v[n_snt:])
            tet.extend(v[n_snt:])

        trs = [items[idx] for idx in trs]
        trt = [items[idx] for idx in trt]
        tet = [items[idx] for idx in tet]

    rng.shuffle(trs)
    rng.shuffle(trt)
    rng.shuffle(tet)

    return trs, trt, tet


def split_office_train_test(cfg, sm=1, seed=0):
    """
    Split training and test set for office dataset
    :param cfg: configuration
    :param sm: sampling method
    :param seed: seed
    :return: configuration
    """
    for k, v in cfg.items():
        _sets, _items = list_images(v['DATA'])
        cfg[k]['CLS'] = _sets
        # split the train and test
        trs, trt, tet = sampling_office(_items, sm, cfg[k]['SRC-N'], cfg[k]['TGT-N'], seed)
        cfg[k]['SRC-TR'], cfg[k]['TGT-TR'], cfg[k]['TGT-TE'] = trs, trt, tet
    return cfg


def sampling_digits(x, y, sm, sn, seed=0):
    """
    Sampling method for digit datasets
    :param x: data
    :param y: label
    :param sm: sampling method
    :param sn: sampling number
    :param seed: random seed
    :return:
    """
    if sn > 0:
        if sm == 0:
            rng = random.Random(seed)
            idx = list(range(0, len(y)))
            rng.shuffle(idx)
            return [x[idx[: min(sn, len(y))]], y[idx[: min(sn, len(y))]]]
        elif sm == 1:
            rng = random.Random(seed)
            # each class has equivalent samples
            classes = np.unique(y)
            classes_counts = {c: sum(y == c) for c in classes}
            classes_idx = {}
            for c in classes:
                classes_idx[c] = np.where(y == c)[0]

            num_class = len(classes)
            num_sample_per_class = sn // num_class

            num_selected = 0
            classes_selected = {}
            # sampling
            for c in classes:
                rng.shuffle(classes_idx[c])
                classes_selected[c] = classes_idx[c][: min(num_sample_per_class, classes_counts[c])]
                num_selected += classes_selected[c]

            idx_selected = np.array([idx for idx in classes_selected.values()]).ravel()

            x = x[idx_selected]
            y = y[idx_selected].ravel().astype('int32')

            return [x, y]
    else:
        return [x, y]


def split_digits_train_test(cfg, src, tgt, sm=1, seed=0):
    """
    Split training and testing set for digit datasets
    :param cfg: configurations
    :param src: source
    :param tgt: target domains
    :param sm: sampling method
    :param seed: random seed
    :return: configuration
    """
    trs_x, trs_y, tes_x, tes_y = load_pkl(cfg[src]['DATA'])
    trt_x, trt_y, tet_x, tet_y = load_pkl(cfg[tgt]['DATA'])

    cfg[src]['TR'] = sampling_digits(trs_x, trs_y, sm, cfg[src]['SRC-N'], seed)
    cfg[tgt]['TR'] = sampling_digits(trt_x, trt_y, sm, cfg[tgt]['TGT-N'], seed)

    cfg[src]['TE'] = [tes_x, tes_y]
    cfg[tgt]['TE'] = [tet_x, tet_y]

    return cfg


def split_digits_train_test_semi(cfg, src, tgt, sm=1, seed=0):
    """
    Split training and testing set for digit datasets
    :param cfg: configurations
    :param src: source
    :param tgt: target domains
    :param sm: sampling method
    :param seed: random seed
    :return: configuration
    """
    trs_x, trs_y, tes_x, tes_y = load_pkl(cfg[src]['DATA'])
    trt_x, trt_y, tet_x, tet_y = load_pkl(cfg[tgt]['DATA'])

    cfg[src]['TR'] = sampling_digits(trs_x, trs_y, sm, cfg[src]['SRC-N'], seed)
    cfg[tgt]['TR'] = sampling_digits(trt_x, trt_y, sm, cfg[tgt]['TGT-N'], seed)
    cfg[tgt]['TR-U'] = [trt_x, trt_y]
    cfg[src]['TE'] = [tes_x, tes_y]
    cfg[tgt]['TE'] = [tet_x, tet_y]

    return cfg


def cal_len(cls_idx_dict):
    length = 0
    for v in cls_idx_dict.values():
        length += len(v)

    return length


def gen_cls_idx_dict(idx_cls_lst):
    cls_idx_dict = {}
    for idx, y in enumerate(idx_cls_lst):
        y = int(y)
        if y in cls_idx_dict:
            cls_idx_dict[y].append(idx)
        else:
            cls_idx_dict[y] = [idx]
    return cls_idx_dict


