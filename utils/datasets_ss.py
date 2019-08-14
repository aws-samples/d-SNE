"""
This scripts create the semi-supervised datasets for digits, office-31, and visda datasets
"""
import os
import random

from mxnet import recordio, image, nd
from mxnet.gluon.data.dataset import Dataset

from utils.io import load_json, save_json
from .datasets_funcs import gen_cls_idx_dict


class DomainRecPairSemiDataset(Dataset):
    def __init__(self, rec_f1, rec_f2, tform1, tform2):
        self.rec_f1 = rec_f1
        self.idx_f1 = os.path.splitext(rec_f1)[0] + '.idx'
        self.tform1 = tform1

        self.rec_f2 = rec_f2
        self.idx_2 = os.path.splitext(rec_f2)[0] + '.idx'
        self.tform2 = tform2

        self._fork()
        self._flag = 1

    def _fork(self):

        self.rec1 = recordio.MXIndexedRecordIO(self.idx_f1, self.rec_f1, 'r')
        self.cls_idx_d1 = self.load_or_gen_dict(self.rec_f1, self.rec1)
        self.idx1 = list(self.rec1.idx.keys())

        self.rec2 = recordio.MXIndexedRecordIO(self.idx_2, self.rec_f2, 'r')
        self.cls_idx_d2 = self.load_or_gen_dict(self.rec_f2, self.rec2)
        self.idx2 = list(self.rec2.idx.keys())

    @staticmethod
    def load_or_gen_dict(rec_f, rec):
        cls_idx_dict_f = os.path.splitext(rec_f)[0] + '.json'
        if os.path.exists(cls_idx_dict_f):
            cls_idx_dict = load_json(cls_idx_dict_f)
            return cls_idx_dict
        else:
            idx_cls_lst = []
            for idx in rec.idx.keys():
                record = rec.read_idx(idx)
                h, _ = recordio.unpack(record)
                idx_cls_lst.append([idx, h.label])

            cls_idx_dict = {}
            for idx, y in idx_cls_lst:
                if y in cls_idx_dict:
                    cls_idx_dict[y].append(idx)
                else:
                    cls_idx_dict[y] = [idx]

            save_json(cls_idx_dict, os.path.splitext(rec_f)[0] + '.json')

            return cls_idx_dict

    def __getitem__(self, idx):
        r1 = self.rec1.read_idx(self.idx1[idx[0]])
        h1, im1 = recordio.unpack(r1)
        im1 = image.imdecode(im1, self._flag)
        l1 = h1.label
        im1 = self.tform1(im1)

        r2 = self.rec2.read_idx(self.idx2[idx[1]])
        h2, im2 = recordio.unpack(r2)
        im2 = image.imdecode(im2, self._flag)

        im2_1 = self.tform2(im2)
        im2_2 = self.tform2(im2)

        return im1, l1, im2_1, im2_2

    def __len__(self):
        return len(self.idx1)

    @staticmethod
    def cal_len(cls_idx_dict):
        length = 0
        for v in cls_idx_dict.values():
            length += len(v)
        return length


class DomainRecPseudoLabelDataset(Dataset):
    def __init__(self, rec_f, labels, tform=None):
        self.rec_f = rec_f
        self.idx_f = os.path.splitext(rec_f)[0] + '.idx'
        self.tform = tform
        self.labels = labels
        self._fork()
        self._flag = 1

    def _fork(self):
        self.rec = recordio.MXIndexedRecordIO(self.idx_f, self.rec_f, 'r')
        self.idx = list(self.rec.idx.keys())

    def __getitem__(self, idx):
        r = self.rec.read_idx(idx)
        h, im = recordio.unpack(r)
        im = image.imdecode(im, self._flag)
        l = h.label

        pseudo_l = self.labels[idx]

        if self.tform is not None:
            im = self.tform(im)

        return im, l, pseudo_l

    def __len__(self):
        return len(self.rec.keys)


class DomainArrayPairSemiDataset(Dataset):
    """
        Domain Array Dataset, designed for digits datasets
        """

    def __init__(self, arr1, arr2, tform1=None, tform2=None, pseudo_labels=None):
        """
        Initialization of dataset
        :param arr1: source array
        :param arr2: target array
        :param tform1: transformers for source array
        :param tform2: transformers for target array
        """
        self.arr1 = arr1
        self.use1 = False if arr1 is None else True

        self.arr2 = arr2
        self.use2 = False if arr2 is None else True

        self.tform1 = tform1
        self.tform2 = tform2

        self.pseudo_labels = pseudo_labels

        self._gen_cls_idx_dicts()

    def _gen_cls_idx_dicts(self):
        if self.use1:
            self.cls_idx_d1 = gen_cls_idx_dict(self.arr1[1])
            self.idx1 = list(range(len(self.arr1[0])))

        if self.use2:
            self.cls_idx_d2 = gen_cls_idx_dict(self.arr2[1])
            self.idx2 = list(range(len(self.arr2[0])))

    def __getitem__(self, idx):
        """
        Override the function getitem
        :param idx: index
        :return:
        """
        if self.use1 and not self.use2:
            im, l = self.arr1[0][idx], self.arr1[1][idx]
            im = nd.array(im, dtype='float32')
            if self.tform1 is not None:
                im = self.tform1(im)

            return im, l
        elif self.use2 and not self.use1:
            im, l = self.arr2[0][idx], self.arr2[1][idx]
            im = nd.array(im, dtype='float32')
            if self.tform2 is not None:
                im = self.tform2(im)

            return im, l
        else:
            idx1, idx2 = idx
            im1, l1 = self.arr1[0][idx1], self.arr1[1][idx1]
            im2, l2 = self.arr2[0][idx2], self.arr2[1][idx2]

            im1 = nd.array(im1, dtype='float32')
            im2 = nd.array(im2, dtype='float32')

            if self.tform1 is not None:
                im1 = self.tform1(im1)

            if self.tform2 is not None:
                im2 = self.tform2(im2)

            lc = 1 if l1 == l2 else 0

            return im1, l1, im2, l2, lc

    def __len__(self):
        if self.use1 and not self.use2:
            return len(self.idx1)
        elif self.use2 and not self.use1:
            return len(self.idx2)
        else:
            return len(self.idx1)


class DomainArrayTripletDataset(Dataset):
    def __init__(self, arr_s, arr_l, arr_u, tform_s=None, tform_l=None, tform_u=None, ratio=1, pseudo_labels=None):
        self.arr_s = arr_s
        self.arr_l = arr_l
        self.arr_u = arr_u

        self._create_labeled_target_cls_idx_dict()

        self.tform_s = tform_s
        self.tform_l = tform_l
        self.tform_u = tform_u

        self._ratio = ratio

        self.pseudo_labels = pseudo_labels

        self._create_unlabeled_target_cls_idx_dict()

        self.length_s = len(self.arr_s[0])
        self.length_l = len(self.arr_l[0])
        self.length_u = len(self.arr_u[0])

    def _create_labeled_target_cls_idx_dict(self):
        cls_idx_dict = {}
        for i, c in enumerate(self.arr_l[1]):
            if c in cls_idx_dict:
                cls_idx_dict[c].append(i)
            else:
                cls_idx_dict[c] = [i]

        self._cls_idx_dict_l = cls_idx_dict

    def _create_unlabeled_target_cls_idx_dict(self):
        if self.pseudo_labels is not None:
            cls_idx_dict = {}
            for i, c in enumerate(self.pseudo_labels):
                if c in cls_idx_dict:
                    cls_idx_dict[c].append(i)
                else:
                    cls_idx_dict[c] = [i]

            self._cls_idx_dict_u = cls_idx_dict

    def ostg(self, l_s):
        id_l = self.ospg(self._cls_idx_dict_l, l_s)

        if self.pseudo_labels is None:
            id_u = random.randint(0, self.length_u-1)
        else:
            l_t = self.arr_l[1][id_l]
            id_u = self.ospg(self._cls_idx_dict_u, l_t)

        return id_l, id_u

    def ospg(self, cls_idx_dict, l):
        rnd = random.uniform(0, 1)
        if rnd > 1. / (1 + self._ratio):
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

    def __getitem__(self, id_s):
        im_s, l_s = self.arr_s[0][id_s], self.arr_s[1][id_s]
        im_s = nd.array(im_s, dtype='float32')

        id_l, id_u = self.ostg(l_s)
        im_l, l_l = self.arr_l[0][id_l], self.arr_l[1][id_l]
        im_l = nd.array(im_l, dtype='float32')

        im_u, l_u = self.arr_u[0][id_u], self.arr_u[1][id_u]
        im_u = nd.array(im_u, dtype='float32')

        if self.tform_s is not None:
            im_s = self.tform_s(im_s)

        if self.tform_l is not None:
            im_l = self.tform_l(im_l)

        if self.tform_u is not None:
            im_u1 = self.tform_l(im_u)
        else:
            im_u1 = im_u

        if self.tform_u is not None:
            im_u2 = self.tform_u(im_u)
        else:
            im_u2 = im_u

        return im_s, l_s, im_l, l_l, im_u1, im_u2

    def __len__(self):
        return self.length_s


class DomainRecTripletDataset(Dataset):
    def __init__(self, rec_fs, rec_fl, rec_fu, tform_s=None, tform_l=None, tform_u=None, ratio=1, pseudo_labels=None):
        self.rec_fs = rec_fs
        self.rec_fl = rec_fl
        self.rec_fu = rec_fu

        self._flag = 1

        self.idx_fs = os.path.splitext(self.rec_fs)[0] + '.idx'
        self.idx_fl = os.path.splitext(self.rec_fl)[0] + '.idx'
        self.idx_fu = os.path.splitext(self.rec_fu)[0] + '.idx'

        self._fork()

        self._create_labeled_target_cls_idx_dict()

        self.tform_s = tform_s
        self.tform_l = tform_l
        self.tform_u = tform_u

        self._ratio = ratio

        self.pseudo_labels = pseudo_labels

        self._create_unlabeled_target_cls_idx_dict()

        self.length_s = len(self.idx_s)
        self.length_l = len(self.idx_l)
        self.length_u = len(self.idx_u)

    def _fork(self):
        self.rec_s = recordio.MXIndexedRecordIO(self.idx_fs, self.rec_fs, 'r')
        self.idx_s = list(self.rec_s.idx.keys())

        self.rec_l = recordio.MXIndexedRecordIO(self.idx_fl, self.rec_fl, 'r')
        self.idx_l = list(self.rec_l.idx.keys())

        self.rec_u = recordio.MXIndexedRecordIO(self.idx_fu, self.rec_fu, 'r')
        self.idx_u = list(self.rec_u.idx.keys())

    @staticmethod
    def load_or_gen_dict(rec_f, rec):
        cls_idx_dict_f = os.path.splitext(rec_f)[0] + '.json'
        if os.path.exists(cls_idx_dict_f):
            cls_idx_dict = load_json(cls_idx_dict_f)
            keys = list(cls_idx_dict.keys())
            for k in keys:
                cls_idx_dict[int(k)] = cls_idx_dict.pop(k)
            return cls_idx_dict
        else:
            idx_cls_lst = []
            for idx in rec.idx.keys():
                record = rec.read_idx(idx)
                h, _ = recordio.unpack(record)
                idx_cls_lst.append([idx, h.label])

            cls_idx_dict = {}
            for idx, y in idx_cls_lst:
                y = int(y)
                if y in cls_idx_dict:
                    cls_idx_dict[y].append(idx)
                else:
                    cls_idx_dict[y] = [idx]

            save_json(cls_idx_dict, os.path.splitext(rec_f)[0] + '.json')

            return cls_idx_dict

    @staticmethod
    def load_or_gen_list(rec_f, rec):
        idx_cls_lst_f = os.path.splitext(rec_f)[0] + '-lst.json'
        if os.path.exists(idx_cls_lst_f):
            idx_cls_lst = load_json(idx_cls_lst_f)
            idx_cls_lst = [int(l) for l in idx_cls_lst]
            return idx_cls_lst
        else:
            idx_cls_lst = []
            for idx in rec.idx.keys():
                record = rec.read_idx(idx)
                h, _ = recordio.unpack(record)
                idx_cls_lst.append(int(h.label))

            save_json(idx_cls_lst, idx_cls_lst_f)

            return idx_cls_lst

    def _create_labeled_target_cls_idx_dict(self):
        self._cls_idx_dict_l = self.load_or_gen_dict(self.rec_fl, self.rec_l)
        self._idx_cls_lst_l = self.load_or_gen_list(self.rec_fl, self.rec_l)

    def _create_unlabeled_target_cls_idx_dict(self):
        if self.pseudo_labels is not None:
            cls_idx_dict = {}
            for i, c in enumerate(self.pseudo_labels):
                c = int(c)
                if c in cls_idx_dict:
                    cls_idx_dict[c].append(i)
                else:
                    cls_idx_dict[c] = [i]

            self._cls_idx_dict_u = cls_idx_dict

    def ostg(self, l_s):
        id_l = self.ospg(self._cls_idx_dict_l, l_s)

        if self.pseudo_labels is None:
            id_u = random.randint(0, self.length_u-1)
        else:
            l_t = self._idx_cls_lst_l[id_l]
            id_u = self.ospg(self._cls_idx_dict_u, l_t)

        return id_l, id_u

    def ospg(self, cls_idx_dict, l):
        rnd = random.uniform(0, 1)
        l = int(l)

        if rnd > 1. / (1 + self._ratio):
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

    @staticmethod
    def read_record(rec, idx):
        record = rec.read_idx(idx)
        header, im = recordio.unpack(record)
        im = image.imdecode(im)

        label = header.label

        return im, label

    def __getitem__(self, id_s):
        im_s, l_s = self.read_record(self.rec_s, id_s)

        id_l, id_u = self.ostg(l_s)

        im_l, l_l = self.read_record(self.rec_l, id_l)

        im_u, l_u = self.read_record(self.rec_u, id_u)

        if self.tform_s is not None:
            im_s = self.tform_s(im_s)

        if self.tform_l is not None:
            im_l = self.tform_l(im_l)

        if self.tform_u is not None:
            im_u1 = self.tform_l(im_u)
        else:
            im_u1 = im_u

        if self.tform_u is not None:
            im_u2 = self.tform_u(im_u)
        else:
            im_u2 = im_u

        return im_s, l_s, im_l, l_l, im_u1, im_u2, l_u

    def __len__(self):
        return self.length_s


class DomainRecTripletDatasetv2(Dataset):
    """
    Labeled target dataset focused data loader
    """
    def __init__(self,  rec_fl, rec_fs, rec_fu, tform_s=None, tform_l=None, tform_u=None, ratio=1, pseudo_labels=None,
                 samples_class=8, num_class=12):

        self.rec_fl = rec_fl
        self.rec_fs = rec_fs
        self.rec_fu = rec_fu

        self._flag = 1

        self.idx_fl = os.path.splitext(self.rec_fl)[0] + '.idx'
        self.idx_fs = os.path.splitext(self.rec_fs)[0] + '.idx'
        self.idx_fu = os.path.splitext(self.rec_fu)[0] + '.idx'

        self._fork()

        self._create_labeled_target_cls_idx_dict()

        self.tform_s = tform_s
        self.tform_l = tform_l
        self.tform_u = tform_u

        self._ratio = ratio

        self.pseudo_labels = pseudo_labels

        self._create_unlabeled_target_cls_idx_dict()

        self.length_s = len(self.idx_s)
        self.length_l = len(self.idx_l)
        self.length_u = len(self.idx_u)

        self.samples_class = samples_class
        self.num_class = num_class
        self.length = (self.num_class // samples_class + 1) * samples_class

    def _fork(self):
        self.rec_s = recordio.MXIndexedRecordIO(self.idx_fs, self.rec_fs, 'r')
        self.idx_s = list(self.rec_s.idx.keys())

        self.rec_l = recordio.MXIndexedRecordIO(self.idx_fl, self.rec_fl, 'r')
        self.idx_l = list(self.rec_l.idx.keys())

        self.rec_u = recordio.MXIndexedRecordIO(self.idx_fu, self.rec_fu, 'r')
        self.idx_u = list(self.rec_u.idx.keys())

    @staticmethod
    def load_or_gen_list_dict(rec_f, rec):
        cls_idx_dict_f = os.path.splitext(rec_f)[0] + '.json'
        idx_cls_lst_f = os.path.splitext(rec_f)[0] + '-lst.json'
        if os.path.exists(cls_idx_dict_f) and os.path.exists(idx_cls_lst_f):
            idx_cls_lst = load_json(idx_cls_lst_f)
            cls_idx_dict = load_json(cls_idx_dict_f)

            idx_cls_lst = [int(item) for item in idx_cls_lst]

            keys = list(cls_idx_dict.keys())
            for k in keys:
                cls_idx_dict[int(k)] = cls_idx_dict.pop(k)
            return idx_cls_lst, cls_idx_dict
        else:
            idx_cls_lst = []
            for idx in rec.idx.keys():
                record = rec.read_idx(idx)
                h, _ = recordio.unpack(record)
                idx_cls_lst.append([idx, int(h.label)])

            cls_idx_dict = {}
            for idx, y in idx_cls_lst:
                if y in cls_idx_dict:
                    cls_idx_dict[y].append(idx)
                else:
                    cls_idx_dict[y] = [idx]

            idx_cls_lst = [int(l) for _, l in idx_cls_lst]

            save_json(idx_cls_lst, idx_cls_lst_f)
            save_json(cls_idx_dict, cls_idx_dict_f)

            return idx_cls_lst, cls_idx_dict

    def _create_labeled_target_cls_idx_dict(self):
        self.idx_cls_lst_l, self.cls_idx_dict_l = self.load_or_gen_list_dict(self.rec_fl, self.rec_l)
        self.idx_cls_lst_s, self.cls_idx_dict_s = self.load_or_gen_list_dict(self.rec_fs, self.rec_s)

    def _create_unlabeled_target_cls_idx_dict(self):
        if self.pseudo_labels is not None:
            cls_idx_dict = {}
            for i, c in enumerate(self.pseudo_labels):
                c = int(c)
                if c in cls_idx_dict:
                    cls_idx_dict[c].append(i)
                else:
                    cls_idx_dict[c] = [i]

            self.cls_idx_dict_u = cls_idx_dict
        else:
            self.cls_idx_dict_u = None

    def read_record(self, rec, idx):
        record = rec.read_idx(idx)
        header, im = recordio.unpack(record)
        im = image.imdecode(im)
        label = header.label

        return im, label

    def __getitem__(self, idx):
        id_l, id_s, id_u = idx

        im_s, l_s = self.read_record(self.rec_s, id_s)
        im_l, l_l = self.read_record(self.rec_l, id_l)
        im_u, l_u = self.read_record(self.rec_u, id_u)

        if self.tform_s is not None:
            im_s = self.tform_s(im_s)

        if self.tform_l is not None:
            im_l = self.tform_l(im_l)

        if self.tform_u is not None:
            im_u1 = self.tform_l(im_u)
        else:
            im_u1 = im_u

        if self.tform_u is not None:
            im_u2 = self.tform_u(im_u)
        else:
            im_u2 = im_u

        return im_s, l_s, im_l, l_l, im_u1, im_u2, l_u

    def __len__(self):
        return self.length
