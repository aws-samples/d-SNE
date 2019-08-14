"""
Generate training source and target domain records
"""

import os
import csv
import random
import argparse


def write_csv(path, lines):
    with open(path, 'w') as f:
        csv_writer = csv.writer(f, delimiter='\t')
        for i, (p, l) in enumerate(lines):
            csv_writer.writerow([i, l, p])


def read_image_list(image_list):
    items = []

    with open(image_list, 'r') as f:
        for item in f:
            p, l = item.split()
            items.append((p, l))

    return items


def sampling_images(items):
    if args.m == 0 and args.n > 0:
        # random shuffle
        rng = random.Random(args.seed)
        rng.shuffle(items)
        return items[: min(args.n, len(items))], items[min(args.n, len(items)):]
    elif args.m == 1 and args.n > 0:
        # balanced sampling
        rng = random.Random(args.seed)

        cls = {}
        for idx, (_, c) in enumerate(items):
            if c in cls:
                cls[c].append(idx)
            else:
                cls[c] = [idx]

        n_cls = args.n // len(cls)

        idx_selected = []
        idx_unselected = []
        # sampling
        for k, v in cls.items():
            rng.shuffle(v)
            idx_selected.extend(v[: min(n_cls, len(v))])
            idx_unselected.extend(v[min(n_cls, len(v)): ])

        train_list = [items[i] for i in idx_selected]
        test_list = [items[i] for i in idx_unselected]

        rng.shuffle(train_list)
        rng.shuffle(test_list)

        return train_list, test_list
    else:
        rng = random.Random(args.seed)
        rng.shuffle(items)
        return items, []


def generate_list():
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    items = read_image_list(args.image_list)
    train_items, test_items = sampling_images(items)

    if len(train_items) > 0:
        write_csv(os.path.join(args.out_dir, '%s-%d.lst' % (args.prefix, args.n)), train_items)

    if len(test_items) > 0:
        write_csv(os.path.join(args.out_dir, '%s-%d.lst' % (args.prefix, len(test_items))), test_items)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('image_list', help='image list')
    parser.add_argument('prefix', help='prefix')
    parser.add_argument('--m', type=int, default=1, help='sampling method')
    parser.add_argument('--n', type=int, default=1200, help='sampling number')
    parser.add_argument('--seed', type=int, default=0, help='seed')
    parser.add_argument('--out-dir', type=str, default='datasets/VisDA17')

    args = parser.parse_args()

    generate_list()
