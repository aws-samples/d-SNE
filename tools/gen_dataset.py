import os
import argparse
import numpy as np
import struct
import pickle as pk


class MNIST:
    def __init__(self, root):
        self.root = root

    def read(self):
        def read_img(path):
            with open(path, 'rb') as fimg:
                magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
                img = np.fromfile(fimg, dtype=np.uint8).reshape(-1, rows, cols)

            return img

        def read_lbl(path):
            with open(path, 'rb') as flbl:
                magic, num = struct.unpack(">II", flbl.read(8))
                lbl = np.fromfile(flbl, dtype=np.int8)

            return lbl

        train_img = read_img(os.path.join(self.root, 'train-images-idx3-ubyte'))
        train_lbl = read_lbl(os.path.join(self.root, 'train-labels-idx1-ubyte'))

        test_img = read_img(os.path.join(self.root, 't10k-images-idx3-ubyte'))
        test_lbl = read_lbl(os.path.join(self.root, 't10k-labels-idx1-ubyte'))

        self.dataset = {'TR': [train_img, train_lbl], 'TE': [test_img, test_lbl]}

    def dump(self):
        with open(os.path.join(self.root, self.__class__.__name__ + '.pkl'), 'wb') as fout:
            pk.dump(self.dataset, fout)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', )
    parser.add_argument('-d', '--dataset', default='mnist', help='dataset')

    args = parser.parse_args()

    if args.dataset.lower() == 'mnist':
        dataset = MNIST(args.dir)
        dataset.read()
        dataset.dump()
    else:
        print('Required download the dataset and packed by yourself, sorry for inconvenience')
