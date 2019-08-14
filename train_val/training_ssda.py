"""
Main training code for semi-supervised domain adaptation
"""

import os
from multiprocessing import cpu_count
import numpy as np
import importlib

from mxnet import autograd, nd
from mxnet.metric import Accuracy, Loss
from mxnet.gluon.loss import SoftmaxCrossEntropyLoss
from mxnet.gluon.data import DataLoader

from utils.io import load_json
from utils.datasets_su import DomainArrayDataset, DomainRecDataset
from utils.datasets_ss import DomainArrayTripletDataset, DomainRecTripletDataset, DomainRecTripletDatasetv2
from utils.datasets_funcs import split_digits_train_test_semi
from utils.samplers import TripletBalancedSampler
from train_val.training_sda import DomainModel
from train_val.custom_layers import dSNELoss, SoftmaxL2Loss


class MeanTeacher(DomainModel):
    def __init__(self, args, use_teacher=True, train_tgt=True):
        self.train_slu_loader = None
        self.use_teacher = use_teacher
        self.train_tgt = train_tgt
        super(MeanTeacher, self).__init__(args)
        self.alpha = self.args.alpha
        self.beta = self.args.beta

    @staticmethod
    def create_metrics():
        """
        Create metrics
        :return: metrics
        """
        metrics = {'Train-Xent-Src': Loss(),
                   'Train-Xent-Tgt-l': Loss(),
                   'Train-Xent-Tgt-Ul': Loss(),
                   'Train-Aux-Src': Loss(),
                   'Train-Aux-Tgt-l': Loss(),
                   'Train-Aux-Tgt-Ul': Loss(),
                   'Train-Cons-Src': Loss(),
                   'Train-Cons-Tgt-l': Loss(),
                   'Train-Cons-Tgt-Ul': Loss(),
                   'Train-Total-Src': Loss(),
                   'Train-Total-Tgt-l': Loss(),
                   'Train-Total-Tgt-Ul': Loss(),
                   'Train-Acc-Src': Accuracy(),
                   'Train-Acc-Tgt-l': Accuracy(),
                   'Train-Acc-Tgt-Ul': Accuracy()}
        return metrics

    def create_inference(self, ema=True):
        if self.args.bb == 'lenetplus' or self.args.bb == 'conv2' or self.args.bb == 'conv13':
            inference = importlib.import_module('models.%s' % self.args.bb).get_inference(
                classes=self.args.nc, embed_size=self.args.embed_size, use_dropout=self.args.dropout,
                use_norm=self.args.l2n, use_bn=self.args.bn, use_inn=self.args.inn)
        elif self.args.bb == 'resnet':
            inference = importlib.import_module('models.%s' % self.args.bb).get_inference(
                num_layers=self.args.nlayers, classes=self.args.nc, embed_size=self.args.embed_size,
                use_dropout=self.args.dropout, thumbnail=False, use_norm=self.args.l2n)
        elif self.args.bb == 'vgg':
            inference = importlib.import_module('models.%s' % self.args.bb).get_inference(
                num_layers=self.args.nlayers, classes=self.args.nc, embed_size=self.args.embed_size,
                use_dropout=self.args.dropout, use_bn=self.args.bn, use_norm=self.args.l2n)
        else:
            raise NotImplementedError

        if self.args.hybridize:
            inference.hybridize()
        self.load_params(inference)

        if ema:
            for params in inference.collect_params():
                inference.collect_params()[params].grad_req = 'null'

        return inference

    def load_digits_cfg(self):
        cfg = load_json(self.args.cfg)
        cfg = split_digits_train_test_semi(cfg, self.args.src.upper(), self.args.tgt.upper(), 1, self.args.seed)

        trs = cfg[self.args.src.upper()]['TR']
        trt_l = cfg[self.args.tgt.upper()]['TR']
        trt_u = cfg[self.args.tgt.upper()]['TR-U']
        tes = cfg[self.args.src.upper()]['TE']
        tet = cfg[self.args.tgt.upper()]['TE']

        return trs, trt_l, trt_u, tes, tet

    def create_digits_datasets(self, train_tforms, eval_tforms):
        """
        Create Digits datasets
        :param train_tforms: training transformers
        :param eval_tforms: evaluation transformers
        :return:
            trs_set: training source set
            trt_set: training target set
            tes_set: testing source set
            tet_set: testing target set
        """
        # Read config
        trs, trt_l, trt_u, tes, tet = self.load_digits_cfg()
        tr_slu_set = DomainArrayTripletDataset(trs, trt_l, trt_u, train_tforms, train_tforms, train_tforms,
                                               self.args.ratio)
        tes_set = DomainArrayDataset(tes, tforms=eval_tforms)
        tet_set = DomainArrayDataset(tet, tforms=eval_tforms)

        return tr_slu_set, tes_set, tet_set

    def create_visda_datasets(self, train_tforms, eval_tforms):
        trs, trt, tes, tet = self.load_visda_cfg()

        tr_slu_set = DomainRecTripletDataset(trs, trt, tet, train_tforms, train_tforms, train_tforms,
                                             self.args.ratio)
        tes_set = DomainRecDataset(tes, tforms=eval_tforms)
        tet_set = DomainRecDataset(tet, tforms=eval_tforms)

        return tr_slu_set, tes_set, tet_set

    def create_loader(self):
        """
        Create data loader
        :return: data loaders
        """
        cpus = cpu_count()
        train_tforms, eval_tforms = self.create_transformer()

        if 'digits' in self.args.cfg:
            tr_slu_set, tes_set, tet_set = self.create_digits_datasets(train_tforms, eval_tforms)
        elif 'visda' in self.args.cfg:
            tr_slu_set, tes_set, tet_set = self.create_visda_datasets(train_tforms, eval_tforms)
        else:
            raise NotImplementedError

        self.train_slu_loader = DataLoader(tr_slu_set, self.args.bs, shuffle=True, num_workers=cpus)
        self.test_src_loader = DataLoader(tes_set, self.args.bs, shuffle=False, num_workers=cpus)
        self.test_tgt_loader = DataLoader(tet_set, self.args.bs, shuffle=False, num_workers=cpus)

    def train(self):
        """
        Training process for Auxiliary model
        :return: None
        """
        # create student inference
        student = self.create_inference(ema=False)
        # create teacher inference
        if self.use_teacher:
            teacher = self.create_inference(ema=True)
        else:
            teacher = None
        # create trainer
        trainer = self.create_trainer(student)

        for cur_epoch in range(self.args.start_epoch + 1, self.args.end_epoch + 1):
            self.cur_epoch = cur_epoch
            # reset metrics
            self.reset_metrics()
            # training
            self.train_epoch(student, trainer, teacher)
            # update learning rate
            self.update_lr(trainer)

        self.logger.update_scalar('Epoch [%d]: Best-Acc' % self.records['Epoch']['Epoch'],
                                  self.records['Epoch']['Tgt-Acc'])
        self.logger.update_scalar('Iter [%d]: Best-Acc' % self.records['Iter']['Iter'],
                                  self.records['Iter']['Tgt-Acc'])

        if self.sw:
            self.sw.close()

    def train_epoch(self, student, trainer, teacher=None):
        """
        Training process for one epoch
        :param student: student inference network
        :param teacher: teacher inference network
        :param trainer: trainer
        :return:
        """
        for xs, ys, xl, yl, xu0, xu1, _ in self.train_slu_loader:
            xs = xs.as_in_context(self.args.ctx[0])
            ys = ys.as_in_context(self.args.ctx[0])
            xl = xl.as_in_context(self.args.ctx[0])
            yl = yl.as_in_context(self.args.ctx[0])
            xu0 = xu0.as_in_context(self.args.ctx[0])
            xu1 = xu1.as_in_context(self.args.ctx[0])

            if self.args.train_src:
                # update with xs focus
                self.train_batch(xs, ys, xl, yl, xu0, xu1, student, teacher, target=False)
                trainer.step(xs.shape[0])

            if self.train_tgt:
                # update with xl focus
                self.train_batch(xl, yl, xs, ys, xu1, xu0, student, teacher, target=True)
                trainer.step(xl.shape[0])

            # exponential moving average for teacher
            if teacher is not None:
                self.update_ema(student, teacher)

            if self.args.log_itv > 0 and self.cur_iter % self.args.log_itv == 0:
                # evaluate
                self.log_iter()
                if self.args.eval:
                    self.eval(student, self.test_tgt_loader, target=True, epoch=False)
                # reset to False
                self.log_src = False

        self.log_epoch()
        if self.args.eval and self.cur_epoch > self.args.eval_epoch:
            self.eval(student, self.test_tgt_loader, target=True, epoch=True)

    def train_batch(self, x0, y0, x1, y1, xu0, xu1, student, teacher=None, target=False):
        criterion_xent = SoftmaxCrossEntropyLoss()
        criterion_consistency = SoftmaxL2Loss()
        postfix = 'Tgt-l' if target else 'Src'

        with autograd.record():
            y0_hat, fss = student(x0)
            # cross entropy
            loss_xent = criterion_xent(y0_hat, y0)

            self.metrics['Train-Xent-%s' % postfix].update(None, [loss_xent])
            self.metrics['Train-Acc-%s' % postfix].update([y0], [y0_hat])

            if teacher is not None:
                ysu_hat, fsu = student(xu0)
                ytu_hat, ftu = teacher(xu1)
                loss_consistency = criterion_consistency(ytu_hat, ysu_hat)

                self.metrics['Train-Cons-%s' % postfix].update(None, [loss_consistency])
            else:
                loss_consistency = 0

            # weighted loss
            consistency_weight = self.update_beta()

            loss = loss_xent + consistency_weight * loss_consistency

            self.metrics['Train-Total-%s' % postfix].update(None, [loss])

            self.cur_iter += 1

            loss.backward()

    def update_ema(self, student, teacher):
        ema_decay = min(1 - 1 / (self.cur_iter + 1), self.args.ema_decay)

        for pt, ps in zip(teacher.collect_params().values(), student.collect_params().values()):
            pt.data()[:] = ema_decay * pt.data() + (1 - ema_decay) * ps.data()

    def update_beta(self):
        if self.args.rampup_epoch > 0:
            current = np.clip(self.cur_epoch, 0.0, self.args.rampup_epoch)
            phase = 1.0 - current / self.args.rampup_epoch
            weight = float(np.exp(-5.0 * phase * phase))
        else:
            weight = 1
        return self.beta * weight

    def update_alpha(self):
        # current = np.clip(self.cur_epoch, 0.0, self.args.rampup_epoch)
        # phase = 1.0 - current / self.args.rampup_epoch
        # weight = float(np.exp(-5.0 * phase * phase))
        return self.alpha

    def log_iter(self):
        if self.args.train_src:
            self.log_file(target=False, epoch=False)

        if self.train_tgt:
            self.log_file(target=True, epoch=False)

    def log_epoch(self):
        if self.args.train_src:
            self.log_file(target=False, epoch=True)

        if self.train_tgt:
            self.log_file(target=True, epoch=True)

    def log_file(self, target=True, epoch=True):
        target_key = 'Tgt-l' if target else 'Src'
        epoch_key = 'Epoch' if epoch else 'Iter'
        record = self.cur_epoch if epoch else self.cur_iter

        self.logger.update_scalar('%s [%d]: Train-XEnt-%s' % (epoch_key, record, target_key),
                                  self.metrics['Train-Xent-%s' % target_key].get()[1])
        # self.logger.update_scalar('%s [%d]: Train-Aux-%s' % (epoch_key, record, target_key),
        #                           self.metrics['Train-Aux-%s' % target_key].get()[1])
        self.logger.update_scalar('%s [%d]: Train-Total-%s' % (epoch_key, record, target_key),
                                  self.metrics['Train-Total-%s' % target_key].get()[1])
        self.logger.update_scalar('%s [%d]: Train-Acc-%s' % (epoch_key, record, target_key),
                                  self.metrics['Train-Acc-%s' % target_key].get()[1])

    def log_record(self, target=True, epoch=True):
        target_key = 'Tgt' if target else 'Src'
        epoch_key = 'Epoch' if epoch else 'Iter'
        record = self.cur_epoch if epoch else self.cur_iter

        self.sw.add_scalar('Loss/Train-%s-XEnt-%s' % (epoch_key, target_key),
                           self.metrics['Train-Xent-%s' % target_key].get()[1],
                           global_step=record)
        self.sw.add_scalar('Acc/Train-%s-Acc-%s' % (epoch_key, target_key),
                           self.metrics['Train-Acc-%s' % target_key].get()[1],
                           global_step=self.cur_epoch)
        self.sw.add_scalar('Loss/Train-%s-Aux-%s' % (epoch_key, target_key),
                           self.metrics['Train-Aux-%s' % target_key].get()[1],
                           global_step=self.cur_epoch)
        self.sw.add_scalar('Acc/Train-%s-Total-%s' % (epoch_key, target_key),
                           self.metrics['Train-Total-%s' % target_key].get()[1],
                           global_step=self.cur_epoch)


class MeanTeacherDSNET(MeanTeacher):
    def __init__(self, args):
        super(MeanTeacherDSNET, self).__init__(args)

    def train_batch(self, x0, y0, x1, y1, xu0, xu1, student, teacher=None, target=False):
        criterion_xent = SoftmaxCrossEntropyLoss()
        criterion_aux = dSNELoss(x0.shape[0], x1.shape[0], self.args.embed_size, self.args.margin, self.args.fn)
        criterion_consistency = SoftmaxL2Loss()
        postfix = 'Tgt-l' if target else 'Src'

        with autograd.record():
            y0_hat, f0 = student(x0)
            y1_hat, f1 = student(x1)
            # cross entropy
            loss_xent = criterion_xent(y0_hat, y0)
            loss_aux = criterion_aux(f0, y0, f1, y1)

            self.metrics['Train-Xent-%s' % postfix].update(None, [loss_xent])
            self.metrics['Train-Aux-%s' % postfix].update(None, [loss_aux])
            self.metrics['Train-Acc-%s' % postfix].update([y0], [y0_hat])

            if teacher is not None:
                ysu_hat, fsu = student(xu0)
                ytu_hat, ftu = teacher(xu1)
                loss_consistency = criterion_consistency(ytu_hat, ysu_hat)

                self.metrics['Train-Cons-%s' % postfix].update(None, [loss_consistency])
            else:
                loss_consistency = 0

            # weighted loss
            aux_weight = self.update_alpha()
            consistency_weight = self.update_beta()

            loss = loss_xent + aux_weight * loss_aux + consistency_weight * loss_consistency

            self.metrics['Train-Total-%s' % postfix].update(None, [loss])

            self.cur_iter += 1

            loss.backward()

    def log_file(self, target=True, epoch=True):
        target_key = 'Tgt-l' if target else 'Src'
        epoch_key = 'Epoch' if epoch else 'Iter'
        record = self.cur_epoch if epoch else self.cur_iter

        self.logger.update_scalar('%s [%d]: Train-XEnt-%s' % (epoch_key, record, target_key),
                                  self.metrics['Train-Xent-%s' % target_key].get()[1])
        # self.logger.update_scalar('%s [%d]: Train-Aux-%s' % (epoch_key, record, target_key),
        #                           self.metrics['Train-Aux-%s' % target_key].get()[1])
        self.logger.update_scalar('%s [%d]: Train-Total-%s' % (epoch_key, record, target_key),
                                  self.metrics['Train-Total-%s' % target_key].get()[1])
        self.logger.update_scalar('%s [%d]: Train-Acc-%s' % (epoch_key, record, target_key),
                                  self.metrics['Train-Acc-%s' % target_key].get()[1])


class MeanTeacherDSNETv2(MeanTeacher):
    def __init__(self, args):
        super(MeanTeacherDSNETv2, self).__init__(args, use_teacher=True)
        self.alpha = self.args.alpha
        self.beta = self.args.beta

    @staticmethod
    def create_metrics():
        """
        Create metrics
        :return: metrics
        """
        metrics = {'Train-Xent-Src': Loss(),
                   'Train-Xent-Tgt': Loss(),
                   'Train-Xent-Tgt-U': Loss(),
                   'Train-Aux-Src': Loss(),
                   'Train-Aux-Tgt': Loss(),
                   'Train-Aux-Tgt-U': Loss(),
                   'Train-Cons-Src': Loss(),
                   'Train-Cons-Tgt': Loss(),
                   'Train-Total-Src': Loss(),
                   'Train-Total-Tgt': Loss(),
                   'Train-Total-Tgt-U': Loss(),
                   'Train-Total': Loss(),
                   'Train-Acc-Src': Accuracy(),
                   'Train-Acc-Tgt': Accuracy(),
                   'Train-Acc-Tgt-U': Accuracy()}
        return metrics

    def create_visda_datasets(self, train_tforms, eval_tforms):
        trs, trt, tes, tet = self.load_visda_cfg()
        if self.args.training:
            pseudo_label = np.loadtxt(os.path.splitext(self.args.model_path)[0] + '-p-label.txt')
        else:
            pseudo_label = None

        tr_slu_set = DomainRecTripletDataset(trs, trt, tet, train_tforms, train_tforms, train_tforms,
                                             pseudo_labels=pseudo_label)
        tes_set = DomainRecDataset(tes, tforms=eval_tforms)
        tet_set = DomainRecDataset(tet, tforms=eval_tforms)

        return tr_slu_set, tes_set, tet_set

    # def create_loader(self):
    #     """
    #     Create data loader
    #     :return: data loaders
    #     """
    #     cpus = cpu_count()
    #     train_tforms, eval_tforms = self.create_transformer()
    #
    #     if 'visda' in self.args.cfg:
    #         tr_slu_set, tes_set, tet_set = self.create_visda_datasets(train_tforms, eval_tforms)
    #     else:
    #         raise NotImplementedError

        # tr_slu_sampler = TripletBalancedSampler(tr_slu_set.idx_cls_lst_l, tr_slu_set.cls_idx_dict_l,
        #                                         tr_slu_set.idx_cls_lst_s, tr_slu_set.cls_idx_dict_s,
        #                                         tr_slu_set.pseudo_labels, tr_slu_set.cls_idx_dict_u,
        #                                         samples_class=4, ratio=self.args.ratio,
        #                                         num_class=self.args.nc)
        # self.train_slu_loader = DataLoader(tr_slu_set, self.args.bs, sampler=tr_slu_sampler, num_workers=cpus)
        # self.train_slu_loader = DataLoader(tr_slu_set, self.args.bs, shuffle=True, num_workers=cpus)
        # self.test_src_loader = DataLoader(tes_set, self.args.bs, shuffle=False, num_workers=cpus)
        # self.test_tgt_loader = DataLoader(tet_set, self.args.bs, shuffle=False, num_workers=cpus)

    def train_epoch(self, student, trainer, teacher=None):
        """
        Training process for one epoch
        :param student: student inference network
        :param teacher: teacher inference network
        :param trainer: trainer
        :return:
        """
        for xs, ys, xl, yl, xu0, xu1, yu in self.train_slu_loader:
            xs = xs.as_in_context(self.args.ctx[0])
            ys = ys.as_in_context(self.args.ctx[0])
            xl = xl.as_in_context(self.args.ctx[0])
            yl = yl.as_in_context(self.args.ctx[0])
            xu0 = xu0.as_in_context(self.args.ctx[0])
            xu1 = xu1.as_in_context(self.args.ctx[0])
            yu = yu.as_in_context(self.args.ctx[0])

            # if self.args.train_src:
            #     # update with xs focus
            #     self.train_batch_su(xs, ys, xl, yl, student, trainer, target=False)
            #
            # if self.train_tgt:
            #     # update with xl focus
            #     self.train_batch_su(xl, yl, xs, ys, student, trainer, target=True)

            self.train_batch_ss(xs, ys, xl, yl, xu0, xu1, yu, student, teacher, trainer)

            # exponential moving average for teacher
            if teacher is not None:
                self.update_ema(student, teacher)

            if self.args.log_itv > 0 and self.cur_iter % self.args.log_itv == 0:
                # evaluate
                self.log_iter()
                if self.args.eval:
                    self.eval(student, self.test_tgt_loader, target=True, epoch=False)
                    if teacher is not None:
                        self.eval(teacher, self.test_tgt_loader, target=True, epoch=False)
                # reset to False
                self.log_src = False

        self.log_epoch()
        if self.args.eval and self.cur_epoch > self.args.eval_epoch:
            self.eval(student, self.test_tgt_loader, target=True, epoch=True)
            if teacher is not None:
                self.eval(teacher, self.test_tgt_loader, target=True, epoch=True)

    # def train_batch_su(self, x1, y1, x2, y2, student, trainer, target=True):
    #     criterion_xent = SoftmaxCrossEntropyLoss()
    #     criterion_aux = dSNETripletLoss(x1.shape[0], x2.shape[0], self.args.embed_size, self.args.margin, self.args.fn)
    #     postfix = 'Tgt' if target else 'Src'
    #
    #     with autograd.record():
    #         y1_hat, f1 = student(x1)
    #         # y2_hat, f2 = student(x2)
    #
    #         loss_xent = criterion_xent(y1_hat, y1)
    #
    #         # w_aux = self.update_alpha()
    #         # loss_aux = criterion_aux(f1, y1, f2, y2)
    #
    #         loss = loss_xent # + w_aux * loss_aux
    #         loss = loss.mean()
    #
    #         self.metrics['Train-Acc-%s' % postfix].update([y1], [y1_hat])
    #         self.metrics['Train-Xent-%s' % postfix].update(None, [loss_xent])
    #         self.metrics['Train-Total'].update(None, [loss])
    #
    #         print(self.metrics['Train-Acc-%s' % postfix].get()[1])
    #
    #         self.cur_iter += 1
    #
    #         loss.backward()
    #
    #     trainer.step(1)

    def train_batch_ss(self, xs, ys, xl, yl, xu0, xu1, yu, student, teacher, trainer):
        criterion_xent = SoftmaxCrossEntropyLoss()
        criterion_aux = dSNELoss(xu0.shape[0], xl.shape[0], self.args.embed_size, self.args.margin, self.args.fn)
        criterion_consistency = SoftmaxL2Loss()

        if self.args.train_src:
            with autograd.record():
                ys_hat, fs = student(xs)
                yl_hat, fl = student(xl)

                loss_xent = criterion_xent(ys_hat, ys)
                loss_aux = criterion_aux(fs, ys, fl, yl)

                loss = loss_xent + 0.1 * loss_aux

                loss.backward()

            trainer.step(xs.shape[0])

        with autograd.record():
            ys_hat, fs = student(xs)
            yl_hat, fl = student(xl)

            loss_xent = criterion_xent(yl_hat, yl)
            loss_aux = criterion_aux(fl, yl, fs, ys)

            loss = loss_xent + 0.1 * loss_aux

            loss.backward()

        trainer.step(xl.shape[0])

        with autograd.record():
            ysu_hat, fsu = student(xu0)
            ytu_hat, ftu = teacher(xu1)

            yl_hat, fl = student(xl)
            ys_hat, fs = student(xs)

            yl_hat.detach()
            fl.detach()

            ys_hat.detach()
            fs.detach()

            p_label, mask = self.pseudo_labeling(ytu_hat, confidence=self.args.thresh)
            loss_xent_u = criterion_xent(ysu_hat, p_label) * mask

            loss_cons = criterion_consistency(ysu_hat, ytu_hat)

            loss_aux = criterion_aux(fsu, p_label, fl, yl) * mask

            # loss = loss_xent_s + loss_xent_l + 10 * loss_xent_u + w_aux * loss_aux + w_cons * loss_cons
            loss = loss_xent_u + 0.1 * loss_cons + 0.001 * loss_aux

            # self.metrics['Train-Acc-Src'].update([ys], [ys_hat])
            # self.metrics['Train-Acc-Tgt'].update([yl], [yl_hat])
            # self.metrics['Train-Xent-Src'].update(None, [loss_xent_s])
            # self.metrics['Train-Xent-Tgt'].update(None, [loss_xent_l])
            self.metrics['Train-Cons-Tgt'].update(None, [loss_cons])
            self.metrics['Train-Acc-Tgt-U'].update([yu], [ysu_hat])
            self.metrics['Train-Xent-Tgt-U'].update(None, loss_xent_u)
            self.metrics['Train-Total'].update(None, [loss])
            # weighted loss
            # aux_weight = self.update_alpha()
            #

            # loss = consistency_weight * loss_consistency + loss_xent_unlabeled

            self.cur_iter += 1

            loss.backward()

        trainer.step(xu0.shape[0])

    def pseudo_labeling(self, logits, confidence=0.):
        softmax = nd.softmax(logits, axis=1)
        prob = nd.max(softmax, axis=1)
        p_label = nd.argmax(softmax, axis=1)
        mask = prob > confidence
        return p_label, mask

    # def update_beta(self):
    #     return self.args.beta

    def log_iter(self):
        # self.logger.update_scalar('%s [%d]: Train-Xent-Src' % ('Iter', self.cur_iter),
        #                           self.metrics['Train-Xent-Src'].get()[1])
        # self.logger.update_scalar('%s [%d]: Train-Xent-Tgt' % ('Iter', self.cur_iter),
        #                           self.metrics['Train-Xent-Tgt'].get()[1])

        # self.logger.update_scalar('%s [%d]: Train-Acc-Src' % ('Iter', self.cur_iter),
        #                           self.metrics['Train-Acc-Src'].get()[1])
        # self.logger.update_scalar('%s [%d]: Train-Acc-Tgt' % ('Iter', self.cur_iter),
        #                           self.metrics['Train-Acc-Tgt'].get()[1])
        self.logger.update_scalar('%s [%d]: Train-Xent-Tgt-U' % ('Iter', self.cur_iter),
                                  self.metrics['Train-Xent-Tgt-U'].get()[1])
        # self.logger.update_scalar('%s [%d]: Train-Cons-Tgt' % ('Iter', self.cur_iter),
        #                           self.metrics['Train-Cons-Tgt'].get()[1])
        self.logger.update_scalar('%s [%d]: Train-Acc-Tgt-U' % ('Iter', self.cur_iter),
                                  self.metrics['Train-Acc-Tgt-U'].get()[1])
        self.logger.update_scalar('%s [%d]: Train-Total' % ('Iter', self.cur_iter),
                                  self.metrics['Train-Total'].get()[1])
        # pass

    def log_epoch(self):
        # self.logger.update_scalar('%s [%d]: Train-Xent-Src' % ('Epoch', self.cur_epoch),
        #                           self.metrics['Train-Xent-Src'].get()[1])
        # self.logger.update_scalar('%s [%d]: Train-Xent-Tgt' % ('Epoch', self.cur_epoch),
        #                           self.metrics['Train-Xent-Tgt'].get()[1])
        # self.logger.update_scalar('%s [%d]: Train-Acc-Src' % ('Epoch', self.cur_epoch),
        #                           self.metrics['Train-Acc-Src'].get()[1])
        # self.logger.update_scalar('%s [%d]: Train-Acc-Tgt' % ('Epoch', self.cur_epoch),
        #                           self.metrics['Train-Acc-Tgt'].get()[1])
        self.logger.update_scalar('%s [%d]: Train-Xent-Tgt-U' % ('Epoch', self.cur_epoch),
                                  self.metrics['Train-Xent-Tgt-U'].get()[1])
        # self.logger.update_scalar('%s [%d]: Train-Cons-Tgt' % ('Epoch', self.cur_epoch),
        #                           self.metrics['Train-Cons-Tgt'].get()[1])
        self.logger.update_scalar('%s [%d]: Train-Acc-Tgt-U' % ('Epoch', self.cur_epoch),
                                  self.metrics['Train-Acc-Tgt-U'].get()[1])
        self.logger.update_scalar('%s [%d]: Train-Total' % ('Epoch', self.cur_epoch),
                                  self.metrics['Train-Total'].get()[1])
        # self.logger.update_scalar('%s [%d]: Train-Acc-Src' % ('Epoch', self.cur_epoch),
        #                           self.metrics['Train-Acc-Src'].get()[1])
        # self.logger.update_scalar('%s [%d]: Train-Acc-Tgt' % ('Epoch', self.cur_epoch),
        #                           self.metrics['Train-Acc-Tgt'].get()[1])
        # self.logger.update_scalar('%s [%d]: Train-Acc-Tgt-U' % ('Epoch', self.cur_epoch),
        #                           self.metrics['Train-Acc-Tgt-U'].get()[1])

    def log_file(self, target=True, epoch=True):
        # target_key = 'Tgt' if target else 'Src'
        # epoch_key = 'Epoch' if epoch else 'Iter'
        # record = self.cur_epoch if epoch else self.cur_iter

        # self.logger.update_scalar('%s [%d]: Train-XEnt-%s' % (epoch_key, record, target_key),
        #                           self.metrics['Train-Xent-%s' % target_key].get()[1])
        # self.logger.update_scalar('%s [%d]: Train-Aux-%s' % (epoch_key, record, target_key),
        #                           self.metrics['Train-Aux-%s' % target_key].get()[1])
        # self.logger.update_scalar('%s [%d]: Train-Total-%s' % (epoch_key, record, target_key),
        #                           self.metrics['Train-Total-%s' % target_key].get()[1])
        # self.logger.update_scalar('%s [%d]: Train-Acc-%s' % (epoch_key, record, target_key),
        #                           self.metrics['Train-Acc-%s' % target_key].get()[1])
        pass
