"""
Sampler functions for the domain adaptation
"""
import random
from mxnet.gluon.data import Sampler

from .datasets_funcs import cal_len
from .io import load_json


class BalancedSampler(Sampler):
    def __init__(self, batch_size, cls_idx_dict):
        """
        Balance Sampler to make sure that every class have similar number of samples in the batch without replacement
        :param batch_size: batch size
        :param cls_idx_dict: class idx dictionary
        """
        self.batch_size = batch_size
        self.cls_idx_dict = cls_idx_dict

        self.n_cls = len(cls_idx_dict.keys())
        self.n_samples = self.batch_size // self.n_cls

        assert self.batch_size >= self.n_cls, "batch size should equal or larger than number of classes"

        self.length = self.cal_len()

    def __iter__(self):
        return iter(self.balance_sampling())

    def balance_sampling(self):
        cls_idx_dict = {}
        for k, v in self.cls_idx_dict.items():
            random.shuffle(v)
            cls_idx_dict[k] = {}
            cls_idx_dict[k]['v'] = v
            cls_idx_dict[k]['p'] = 0

        seq = []

        while len(seq) < self.length:
            for k, v in cls_idx_dict.items():
                m_pointer = min(v['p'] + self.n_samples, len(v['v']))
                seq.extend(v['v'][v['p']: m_pointer])
                cls_idx_dict[k]['p'] = m_pointer

        return seq

    def cal_len(self):
        length = 0
        for v in self.cls_idx_dict.values():
            length += len(v)

        return length

    def __len__(self):
        return self.length


class TwoStreamBalancedSampler(Sampler):
    def __init__(self, batch_size, cls_idx_dict1, cls_idx_dict2, ratio=1):
        """
        Balanced Two Steam Sampler, use cls_idx_dict1 as main dictinary and list
        :param batch_size: batch size
        :param cls_idx_dict1: class index dictionary
        :param cls_idx_dict2: class index dictionary
        :param ratio: negative / positive flag
        """
        self.batch_size = batch_size
        self.cls_idx_dict1 = cls_idx_dict1
        self.cls_idx_dict2 = cls_idx_dict2
        self.ratio = ratio

        assert set(cls_idx_dict1.keys()) == set(cls_idx_dict2.keys()), 'The labels of two classes are not consistent'

        self.n_cls = len(cls_idx_dict1.keys())
        self.n_samples = self.batch_size // self.n_cls

        assert self.batch_size >= self.n_cls, "batch size should equal or larger than number of classes"

        self.length = self.cal_len()

    def balance_sampling(self):
        cls_idx_dict = {}
        for k, v in self.cls_idx_dict1.items():
            random.shuffle(v)
            cls_idx_dict[k] = {}
            cls_idx_dict[k]['v'] = v
            cls_idx_dict[k]['p'] = 0

        seq = []
        cls = []
        while len(seq) < self.length:
            for k, v in cls_idx_dict.items():
                m_pointer = min(v['p'] + self.n_samples, len(v['v']))
                seq.extend(v['v'][v['p']: m_pointer])
                cls.extend(list((k, )*(m_pointer-v['p'])))
                cls_idx_dict[k]['p'] = m_pointer

        return seq, cls

    def cal_len(self):
        length = 0
        for v in self.cls_idx_dict1.values():
            length += len(v)

        return length

    def ospg(self, idx_seq, cls_seq):
        # online sampling pair generation
        pairs = []
        for idx, cls in zip(idx_seq, cls_seq):

            rnd = random.uniform(0, 1)
            if rnd > 1. / (1 + self.ratio):
                # random select class
                cls_set = set(self.cls_idx_dict2.keys())
                cls_set.remove(cls)
                idy = random.randint(0, len(cls_set) - 1)
                yt = list(cls_set)[idy]
                # random select the negative samples
                idy = random.randint(0, len(self.cls_idx_dict2[yt]) - 1)
                idy = self.cls_idx_dict2[yt][idy]
            else:
                idy = random.randint(0, len(self.cls_idx_dict2[cls]) - 1)
                idy = self.cls_idx_dict2[cls][idy]

            pairs.append([idx, idy])

        return pairs

    def __iter__(self):
        idx_seq, cls_seq = self.balance_sampling()
        pairs = self.ospg(idx_seq, cls_seq)
        return iter(pairs)

    def __len__(self):
        return self.length


class TwoStreamBalancedPredsSampler(Sampler):
    def __init__(self, batch_size, cls_idx_dict1, idx_lst2, ratio=1, preds_f=None):
        """
        Two Steam weighted balanced Sampler
        :param batch_size: batch size (bs, )
        :param cls_idx_dict1: class index dictionary (C, )
        :param idx_lst2: index list, [l]
        :param ratio: ratio
        :param preds_f: prediction file, [l x C]
        """
        self.batch_size = batch_size
        self.cls_idx_dict1 = cls_idx_dict1
        self.idx_lst2 = idx_lst2
        self.ratio = ratio
        self.preds_f = preds_f

        self.cls = cls_idx_dict1.keys()
        self.n_cls = len(cls_idx_dict1.keys())
        self.n_samples = self.batch_size // self.n_cls

        assert self.batch_size >= self.n_cls, "batch size should equal or larger than number of classes"

        preds_cls_idx = load_json(self.preds_f)
        keys = list(preds_cls_idx.keys())
        for k in keys:
            preds_cls_idx[int(k)] = preds_cls_idx.pop(k)

        self.preds_cls_idx = preds_cls_idx

        self.length = cal_len(self.cls_idx_dict1)

    def balance_sampling(self):
        cls_idx_dict = {}
        for k, v in self.cls_idx_dict1.items():
            random.shuffle(v)
            cls_idx_dict[k] = {}
            cls_idx_dict[k]['v'] = v
            cls_idx_dict[k]['p'] = 0

        seq = []
        cls = []
        while len(seq) < self.length:
            for k, v in cls_idx_dict.items():
                m_pointer = min(v['p'] + self.n_samples, len(v['v']))
                seq.extend(v['v'][v['p']: m_pointer])
                cls.extend(list((k, )*(m_pointer-v['p'])))
                cls_idx_dict[k]['p'] = m_pointer

        return seq, cls

    def osp2g(self, idx_seq, cls_seq, pred_cls_idx):
        # online sampling with prediction pair generation
        pairs = []
        for idx, cls in zip(idx_seq, cls_seq):
            cls = int(cls)
            rnd = random.uniform(0, 1)
            if rnd > 1. / (1 + self.ratio):
                # negative samples
                # random select class
                cls_set = set(pred_cls_idx.keys())
                cls_set.remove(cls)
                idy = random.randint(0, len(cls_set) - 1)
                yt = list(cls_set)[idy]
                # random select the negative samples
                idy = random.randint(0, len(pred_cls_idx[yt]) - 1)
                idy = pred_cls_idx[yt][idy]
            else:
                # positive samples
                idy = random.randint(0, len(pred_cls_idx[cls]) - 1)
                idy = pred_cls_idx[cls][idy]

            pairs.append([idx, idy])

        return pairs

    def __iter__(self):
        idx_seq, cls_seq = self.balance_sampling()
        pairs = self.osp2g(idx_seq, cls_seq, self.preds_cls_idx)
        return iter(pairs)

    def __len__(self):
        return self.length


class TwoStreamSampler(Sampler):
    def __init__(self, idx1, idx2):
        """
        Two Steam Random Sampler
        :param idx1: index 1
        :param idx2: index 2
        """
        self.lst1 = idx1
        self.lst2 = idx2

        self.length = len(idx1)

    def __iter__(self):
        random.shuffle(self.lst1)
        pairs = []
        for id1 in self.lst1:
            id2 = random.randint(0, len(self.lst2)-1)
            pairs.append([id1, id2])

        return iter(pairs)

    def __len__(self):
        return self.length


class TripletBalancedSampler(Sampler):
    def __init__(self, idx_cls_lst_l, cls_idx_dict_l, idx_cls_lst_s, cls_idx_dict_s,
                 idx_cls_lst_u=None, cls_idx_dict_u=None, len_u=None, samples_class=8, ratio=1, num_class=12):
        self.idx_cls_lst_l = idx_cls_lst_l
        self.cls_idx_dict_l = cls_idx_dict_l

        self.idx_cls_lst_s = idx_cls_lst_s
        self.cls_idx_dict_s = cls_idx_dict_s

        self.idx_cls_lst_u = idx_cls_lst_u
        self.cls_idx_dict_u = cls_idx_dict_u

        self.samples_class = samples_class
        self.num_class = num_class
        self.ratio = ratio
        self.len_u = len_u

        assert idx_cls_lst_u is None or len_u is None, "one of idx_cls_lst_u or len_u should not be None"

        self.length = (self.num_class // samples_class + 1) * samples_class

    def random_sampler(self):
        idx_l = []
        num_loop = self.length // self.num_class + 1

        for _ in range(num_loop):
            keys = list(self.cls_idx_dict_l.keys())
            random.shuffle(keys)

            for k in keys:
                idx_sel = random.sample(self.cls_idx_dict_l[k], self.samples_class)
                idx_l.extend(idx_sel)

        return idx_l

    def ostg(self, idl):
        ids = self.ospg(self.cls_idx_dict_s, idl)

        if self.idx_cls_lst_u is None:
            idu = random.randint(0, self.len_u - 1)
        else:
            idu = self.ospg(self.cls_idx_dict_u, idl)

        return ids, idu

    def ospg(self, cls_idx_dict, l):
        rnd = random.uniform(0, 1)
        l = int(l)

        if rnd > 1. / (1 + self.ratio):
            # random select class
            cls_set = set(cls_idx_dict.keys())
            cls_set.remove(l)
            idx = random.randint(0, len(cls_set) - 1)
            ys = list(cls_set)[idx]
            # random select the negative samples
            idx = random.randint(0, len(cls_idx_dict[ys]) - 1)
            idx = cls_idx_dict[ys][idx]
        else:
            idx = random.randint(0, len(cls_idx_dict[l]) - 1)
            idx = cls_idx_dict[l][idx]

        return idx

    def __iter__(self):
        idx_l = self.random_sampler()

        pairs = []
        for idl in idx_l:
            l = self.idx_cls_lst_l[idl]
            ids, idu = self.ostg(l)
            pairs.append([idl, ids, idu])

        return iter(pairs)

    def __len__(self):
        return self.length
