"""
This script creates supervised dataset class for Digits, office 31 and VisDA dataset
"""
import sys
import os
import random
import json

from mxnet import image, recordio, nd
from mxnet.gluon.data import Dataset
from .io import load_json
from .datasets_funcs import gen_cls_idx_dict


class DomainArrayDataset(Dataset):
    """
    Domain Array Dataset, designed for digits datasets
    """
    def __init__(self, arrs=None, arrt=None, tforms=None, tformt=None, ratio=0):
        """
        Initialization of dataset
        :param arrs: source array
        :param arrt: target array
        :param tforms: transformers for source array
        :param tformt: transformers for target array
        :param ratio: negative/positive ratio
        """
        assert arrs is not None or arrt is not None, "One of src array or tgt array should not be None"

        self.arrs = arrs
        self.use_src = False if arrs is None else True

        self.arrt = arrt
        self.use_tgt = False if arrt is None else True

        self.tforms = tforms
        self.tformt = tformt

        self.ratio = ratio

        if self.use_src and self.use_tgt:
            self.pairs = self._create_pairs()
        elif self.use_src and not self.use_tgt:
            self.pairs = list(range(len(self.arrs[0])))
        elif not self.use_src and self.use_tgt:
            self.pairs = list(range(len(self.arrs[0])))
        else:
            sys.exit("Need to input one source")

    def _create_pairs(self):
        """
        Create pairs for array
        :return:
        """
        pos_pairs, neg_pairs = [], []
        for ids, ys in enumerate(self.arrs[1]):
            for idt, yt in enumerate(self.arrt[1]):
                if ys == yt:
                    pos_pairs.append([ids, ys, idt, yt, 1])
                else:
                    neg_pairs.append([ids, ys, idt, yt, 0])

        if self.ratio > 0:
            random.shuffle(neg_pairs)
            pairs = pos_pairs + neg_pairs[: self.ratio * len(pos_pairs)]
        else:
            pairs = pos_pairs + neg_pairs

        random.shuffle(pairs)

        return pairs

    def __getitem__(self, idx):
        """
        Override the function getitem
        :param idx: index
        :return:
        """
        if self.use_src and not self.use_tgt:
            im, l = self.arrs[0][idx], self.arrs[1][idx]
            im = nd.array(im, dtype='float32')
            if self.tforms is not None:
                im = self.tforms(im)

            return im, l
        elif self.use_tgt and not self.use_src:
            im, l = self.arrt[0][idx], self.arrt[1][idx]
            im = nd.array(im, dtype='float32')
            if self.tformt is not None:
                im = self.tformt(im)

            return im, l
        else:
            [ids, ys, idt, yt, lc] = self.pairs[idx]
            ims, ls = self.arrs[0][ids], self.arrs[1][ids]
            imt, lt = self.arrt[0][idt], self.arrt[1][idt]

            ims = nd.array(ims, dtype='float32')
            imt = nd.array(imt, dtype='float32')

            assert ys == ls
            assert yt == lt

            if self.tforms is not None:
                ims = self.tforms(ims)

            if self.tformt is not None:
                imt = self.tformt(imt)

            return ims, ls, imt, lt, lc

    def __len__(self):
        return len(self.pairs)


class DomainFolderDataset(Dataset):
    """
    Domain Image Folder Dataset, designed for image folder for small dataset such as office31
    """
    def __init__(self, lsts=None, lstt=None, tforms=None, tformt=None, ratio=0):
        """
        Initialization
        :param lsts: source image list (path, id)
        :param lstt: target image list (path, id)
        :param tforms: transformers for source domain
        :param tformt: transformers for target domain
        :param ratio: negative/positive ratio
        """
        assert lsts is not None or lstt is not None, "One of src list or tgt list should not be None"
        # list images
        self.lsts = lsts
        self.use_src = False if lsts is None else True

        self.lstt = lstt
        self.use_tgt = False if lstt is None else True

        # transformer
        self._flag = 1
        self.tforms = tforms
        self.tformt = tformt

        # create items list
        self.ratio = ratio
        if self.use_src and self.use_tgt:
            self.items = self._create_pairs()
        elif self.use_src and not self.use_tgt:
            self.items = self.lsts
        elif not self.use_src and self.use_tgt:
            self.items = self.lstt
        else:
            sys.exit("Need to input one source")

    def _create_pairs(self):
        """
        Create pair of images
        :return:
        """
        pos_pairs, neg_pairs = [], []
        for ps, ls in self.lsts:
            for pt, lt in self.lstt:
                if ls == lt:
                    pos_pairs.append([ps, ls, pt, lt, 1])
                else:
                    neg_pairs.append([ps, ls, pt, lt, 0])

        if self.ratio > 0:
            random.shuffle(neg_pairs)
            sel_pairs = pos_pairs + neg_pairs[: min(len(neg_pairs), self.ratio * len(pos_pairs))]
        else:
            sel_pairs = pos_pairs + neg_pairs

        # shuffle
        random.shuffle(sel_pairs)

        return sel_pairs

    def __getitem__(self, idx):
        if self.use_src and self.use_tgt:
            ims = image.imread(self.items[idx][0], self._flag)
            ls = self.items[idx][1]
            if self.tforms is not None:
                ims = self.tforms(ims)

            imt = image.imread(self.items[idx][2], self._flag)
            lt = self.items[idx][3]
            if self.tformt is not None:
                imt = self.tformt(imt)
            lc = self.items[idx][4]
            return ims, ls, imt, lt, lc
        elif self.use_src and not self.use_tgt:
            im = image.imread(self.items[idx][0], self._flag)
            l = self.items[idx][1]
            if self.tforms is not None:
                im = self.tforms(im)

            return im, l

        elif self.use_tgt and not self.use_src:
            im = image.imread(self.items[idx][0], self._flag)
            l = self.items[idx][1]
            if self.tformt is not None:
                im = self.tformt(im)

            return im, l

        else:
            raise NotImplementedError

    def __len__(self):
        return len(self.items)


class DomainRecDataset(Dataset):
    """
    Domain Image Record file Dataset, designed for large record files dataset, such as VisDA17
    """
    def __init__(self, rec_fs=None, rec_ft=None, tforms=None, tformt=None, ratio=1):
        """
        Initialization for domain record dataset
        :param rec_fs: source record file
        :param rec_ft: target record file
        :param tforms: transformer for source domain
        :param tformt: transformer for target domain
        :param ratio: negative/positive pairs
        """
        # list images
        assert rec_fs is not None or rec_ft is not None, "One of Rec src or Rec tgt should not be None"

        self.rec_fs = rec_fs
        self.use_src = False if rec_fs is None else True
        if self.use_src:
            self.idx_fs = os.path.splitext(rec_fs)[0] + '.idx'
        self.tforms = tforms

        self.rec_ft = rec_ft
        self.use_tgt = False if rec_ft is None else True
        if self.use_tgt:
            self.idx_ft = os.path.splitext(rec_ft)[0] + '.idx'
        self.tformt = tformt

        self._fork()
        self._flag = 1
        self._ratio = ratio

    def _fork(self):
        if self.use_src:
            self.recs = recordio.MXIndexedRecordIO(self.idx_fs, self.rec_fs, 'r')
            self.idxs = list(self.recs.idx.keys())

        if self.use_tgt:
            self.rect = recordio.MXIndexedRecordIO(self.idx_ft, self.rec_ft, 'r')
            self.idxt = list(self.rect.idx.keys())

            if self.use_src:
                cls_lst = []
                for idx in self.idxt:
                    record = self.rect.read_idx(idx)
                    h, _ = recordio.unpack(record)
                    cls_lst.append(h.label)

                self.idxt_cls = self.generate_cls_dict(cls_lst)

    @staticmethod
    def generate_cls_dict(cls_lst):
        cls_dict = {}
        for idx, y in enumerate(cls_lst):
            if y in cls_dict:
                cls_dict[y].append(idx)
            else:
                cls_dict[y] = [idx]

        return cls_dict

    def __getitem__(self, idx):

        if self.use_src and not self.use_tgt:
            record = self.recs.read_idx(self.idxs[idx])
            header, im = recordio.unpack(record)
            im = image.imdecode(im, self._flag)
            l = header.label
            if self.tforms is not None:
                im = self.tforms(im)

            return im, l
        elif self.use_tgt and not self.use_src:
            record = self.rect.read_idx(self.idxt[idx])
            header, im = recordio.unpack(record)
            im = image.imdecode(im, self._flag)
            l = header.label
            if self.tformt is not None:
                im = self.tformt(im)

            return im, l
        else:
            # online sample pairs generation
            records = self.recs.read_idx(self.idxs[idx])
            hs, ims = recordio.unpack(records)
            ims = image.imdecode(ims, self._flag)
            ls = hs.label
            if self.tforms is not None:
                ims = self.tforms(ims)

            rnd = random.uniform(0, 1)
            if rnd > 1. / (1 + self._ratio):
                # random select class
                cls_set = set(self.idxt_cls.keys())
                cls_set.remove(ls)
                idx = random.randint(0, len(cls_set) - 1)
                ys = list(cls_set)[idx]
                # random select the negative samples
                idx = random.randint(0, len(self.idxt_cls[ys]) - 1)
                idx = self.idxt_cls[ys][idx]
            else:
                idx = random.randint(0, len(self.idxt_cls[ls]) - 1)
                idx = self.idxt_cls[ls][idx]

            recordt = self.rect.read_idx(self.idxt[idx])
            ht, imt = recordio.unpack(recordt)
            imt = image.imdecode(imt, self._flag)
            lt = ht.label

            if self.tformt is not None:
                imt = self.tformt(imt)

            yc = 1 if ls == lt else 0

            return ims, ls, imt, lt, yc

    def __len__(self):
        if self.use_src and not self.use_tgt:
            return len(self.idxs)
        elif self.use_tgt and not self.use_src:
            return len(self.idxt)
        else:
            return len(self.idxs)


class DomainArrayPairDataset(Dataset):
    """
    Domain Array Dataset, designed for digits datasets
    """
    def __init__(self, arr1=None, arr2=None, tform1=None, tform2=None):
        """
        Initialization of dataset
        :param arr1: source array
        :param arr2: target array
        :param tform1: transformers for source array
        :param tform2: transformers for target array
        """
        assert arr1 is not None or arr2 is not None, "One of src array or tgt array should not be None"

        self.arr1 = arr1
        self.use1 = False if arr1 is None else True

        self.arr2 = arr2
        self.use2 = False if arr2 is None else True

        self.tform1 = tform1
        self.tform2 = tform2

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


class DomainFolderPairDataset(Dataset):
    def __init__(self, lst1=None, lst2=None, tform1=None, tform2=None):
        """
        Initialization
        :param lst1: source image list (path, id)
        :param lst2: target image list (path, id)
        :param tform1: transformers for source domain
        :param tform2: transformers for target domain
        """
        assert lst1 is not None or lst2 is not None, "One of src list or tgt list should not be None"
        # list images
        self.lst1 = lst1
        self.use1 = False if lst1 is None else True

        self.lst2 = lst2
        self.use2 = False if lst2 is None else True

        # transformer
        self._flag = 1
        self.tform1 = tform1
        self.tform2 = tform2

        # create items list
        self._gen_cls_idx_dicts()

    def _gen_cls_idx_dicts(self):
        if self.use1:
            idx_cls_lst1 = [int(l) for _, l in self.lst1]
            self.cls_idx_d1 = gen_cls_idx_dict(idx_cls_lst1)
            self.idx1 = list(range(len(idx_cls_lst1)))

        if self.use2:
            idx_cls_lst2 = [int(l) for _, l in self.lst2]
            self.cls_idx_d2 = gen_cls_idx_dict(idx_cls_lst2)
            self.idx2 = list(range(len(idx_cls_lst2)))

    def __getitem__(self, idx):
        if self.use1 and self.use2:
            idx1, idx2 = idx
            im1 = image.imread(self.lst1[idx1][0], self._flag)
            l1 = self.lst1[idx1][1]
            if self.tform1 is not None:
                im1 = self.tform1(im1)

            im2 = image.imread(self.lst2[idx2][0], self._flag)
            l2 = self.lst2[idx2][1]
            if self.tform2 is not None:
                im2 = self.tform2(im2)
            lc = 1 if l1 == l2 else 0
            return im1, l1, im2, l2, lc

        elif self.use1 and not self.use2:
            im = image.imread(self.lst1[idx][0], self._flag)
            l = self.lst1[idx][1]
            if self.tform1 is not None:
                im = self.tform1(im)

            return im, l

        elif self.use2 and not self.use1:
            im = image.imread(self.lst2[idx][0], self._flag)
            l = self.lst2[idx][1]
            if self.tform2 is not None:
                im = self.tform2(im)

            return im, l

        else:
            raise NotImplementedError

    def __len__(self):
        return len(self.items)


class DomainRecPairDataset(Dataset):
    def __init__(self, rec_f1=None, rec_f2=None, tform1=None, tform2=None):
        assert rec_f1 is not None or rec_f2 is not None, "One of Rec src or Rec tgt should not be None"

        self.rec_f1 = rec_f1
        self.use1 = False if rec_f1 is None else True
        if self.use1:
            self.idx_f1 = os.path.splitext(rec_f1)[0] + '.idx'
        self.tform1 = tform1

        self.rec_f2 = rec_f2
        self.use2 = False if rec_f2 is None else True
        if self.use2:
            self.idx_2 = os.path.splitext(rec_f2)[0] + '.idx'
        self.tform2 = tform2

        self._fork()
        self._flag = 1

    def _fork(self):
        if self.use1:
            self.rec1 = recordio.MXIndexedRecordIO(self.idx_f1, self.rec_f1, 'r')
            self.cls_idx_d1 = self.load_or_gen_dict(self.rec_f1, self.rec1)
            self.idx1 = list(self.rec1.idx.keys())

        if self.use2:
            self.rec2 = recordio.MXIndexedRecordIO(self.idx_2, self.rec_f2, 'r')
            self.cls_idx_d2 = self.load_or_gen_dict(self.rec_f2, self.rec2)
            self.idx2 = list(self.rec2.idx.keys())

    @staticmethod
    def load_or_gen_dict(rec_f, rec):
        cls_idx_dict_f = os.path.splitext(rec_f)[0] + '.json'
        if os.path.exists(cls_idx_dict_f):
            cls_idx_dict = load_json(cls_idx_dict_f)
            keys = list(cls_idx_dict.keys())
            for k in keys:
                cls_idx_dict[int(float(k))] = cls_idx_dict.pop(k)
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

            with open(os.path.splitext(rec_f)[0] + '.json', 'w') as f:
                json.dump(cls_idx_dict, f, indent=4, sort_keys=True)

            return cls_idx_dict

    def __getitem__(self, idx):
        if self.use1 and not self.use2:
            record = self.rec1.read_idx(self.idx1[idx])
            h, im = recordio.unpack(record)
            im = image.imdecode(im, self._flag)
            l = h.label
            if self.tform1 is not None:
                im = self.tform1(im)

            return im, l
        elif self.use2 and not self.use1:
            record = self.rec1.read_idx(self.idx2[idx])
            h, im = recordio.unpack(record)
            im = image.imdecode(im, self._flag)
            l = h.label
            if self.tform2 is not None:
                im = self.tform2(im)

            return im, l
        else:
            # online sample pairs generation
            r1 = self.rec1.read_idx(self.idx1[idx[0]])
            h1, im1 = recordio.unpack(r1)
            im1 = image.imdecode(im1, self._flag)
            l1 = h1.label
            if self.tform1 is not None:
                im1 = self.tform1(im1)

            r2 = self.rec2.read_idx(self.idx2[idx[1]])
            h2, im2 = recordio.unpack(r2)
            im2 = image.imdecode(im2, self._flag)
            l2 = h2.label

            if self.tform2 is not None:
                im2 = self.tform2(im2)

            yc = 1 if l1 == l2 else 0

            return im1, l1, im2, l2, yc

    def __len__(self):
        if self.use1 and not self.use2:
            return len(self.idx1)
        elif self.use2 and not self.use1:
            return len(self.idx2)
        else:
            return len(self.idx1)

    @staticmethod
    def cal_len(cls_idx_dict):
        length = 0
        for v in cls_idx_dict.values():
            length += len(v)
        return length
