"""
Generate cls dictionary for the rec files
"""

import os
import json
import argparse
from tqdm import tqdm
from mxnet import recordio


def gen_cls_dict():
    rec = recordio.MXIndexedRecordIO(os.path.splitext(args.rec)[0] + '.idx', args.rec, 'r')
    cls_lst = []
    pbar = tqdm(total=len(rec.idx.keys()))
    for idx in rec.idx.keys():
        record = rec.read_idx(idx)
        h, _ = recordio.unpack(record)
        cls_lst.append(int(h.label))
        pbar.update()

    pbar.close()

    cls_dict = {}
    for idx, y in enumerate(cls_lst):
        if y in cls_dict:
            cls_dict[y].append(idx)
        else:
            cls_dict[y] = [idx]

    with open(os.path.splitext(args.rec)[0] + '.json', 'w') as f:
        json.dump(cls_dict, f, indent=4, sort_keys=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('rec', help='rec file')

    args = parser.parse_args()

    gen_cls_dict()
