"""
Generate class index dictionary from predictions
"""
import numpy as np
import argparse

from utils.io import save_json


def gen_cls_idx_dict():
    logits = np.loadtxt(args.preds)
    if len(logits.shape) == 2:
        preds = np.argmax(logits, axis=1)
    else:
        preds = logits

    cls_idx_dict = {}
    for i, y_hat in enumerate(preds):
        if y_hat in cls_idx_dict:
            cls_idx_dict[int(y_hat)].append(i)
        else:
            cls_idx_dict[int(y_hat)] = [i]

    save_json(cls_idx_dict, args.out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('preds', help='predictions')
    parser.add_argument('out', help='out file predictions')

    args = parser.parse_args()

    gen_cls_idx_dict()
