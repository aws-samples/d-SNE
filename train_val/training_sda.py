"""
Main training and evaluation code for domain adaptation
"""

import os
from multiprocessing import cpu_count
import numpy as np
import importlib

from mxnet import autograd, initializer
from mxnet.metric import Loss, Accuracy
from mxnet.gluon import Trainer
from mxnet.gluon.loss import SoftmaxCrossEntropyLoss
from mxnet.gluon.data import DataLoader
from mxnet.gluon.data.vision import transforms
from mxnet.gluon.utils import split_and_load

from utils.datasets_su import DomainFolderDataset, DomainRecDataset, DomainArrayDataset
from utils.datasets_funcs import split_office_train_test, split_digits_train_test
from utils.logging import Logger
from utils.plotting import cal_tsne_embeds, cal_tsne_embeds_src_tgt
from utils.mxnet_utils import MultiEpochScheduler
from utils.io import load_json
from .custom_layers import ContrastiveLoss, dSNELoss


class DomainModel(object):
    """
    Basic Domain Adaption model
    """
    def __init__(self, args):
        """
        Initialize the model
        :param args: user arguments from parser
        """
        self.args = args
        # logger
        self.logger, self.sw = self.create_logger()
        self.log_src, self.log_tgt = False, False
        self.label_dict = None
        # train loader
        self.train_src_loader, self.train_tgt_loader, self.test_src_loader, self.test_tgt_loader = (None,) * 4
        self.create_loader()
        # metrics
        self.metrics = self.create_metrics()
        # learning rate
        self.lr_schlr = None
        self.create_lr_scheduler()
        # records
        self.records = None
        self.reset_record()
        self.cur_epoch = 0
        self.cur_iter = 0

    def create_logger(self):
        """
        Create the logger including the file log and summary log
        :return: logger and summary writer
        """
        if self.args.training:
            logger = Logger(self.args.log, '%s-%s' % (self.args.method, self.args.postfix),
                            rm_exist=self.args.start_epoch == 0)
            logger.update_dict(vars(self.args))

            if self.args.mxboard:
                from mxboard import SummaryWriter
                sw = SummaryWriter(logdir=self.args.log)
            else:
                sw = None
        else:
            logger, sw = None, None

        return logger, sw

    def create_transformer(self):
        train_tforms, eval_tforms = [transforms.Resize(self.args.resize)], [transforms.Resize(self.args.resize)]

        if self.args.random_crop:
            train_tforms.append(transforms.RandomResizedCrop(self.args.size, scale=(0.8, 1.2)))
        else:
            train_tforms.append(transforms.CenterCrop(self.args.size))

        eval_tforms.append(transforms.CenterCrop(self.args.size))

        if self.args.flip:
            train_tforms.append(transforms.RandomFlipLeftRight())

        if self.args.random_color:
            train_tforms.append(transforms.RandomColorJitter(self.args.color_jitter, self.args.color_jitter,
                                                             self.args.color_jitter, 0.1))

        train_tforms.extend([transforms.ToTensor(), transforms.Normalize(self.args.mean, self.args.std)])
        eval_tforms.extend([transforms.ToTensor(), transforms.Normalize(self.args.mean, self.args.std)])

        train_tforms = transforms.Compose(train_tforms)
        eval_tforms = transforms.Compose(eval_tforms)

        return train_tforms, eval_tforms

    def create_loader(self):
        """
        Create data loader
        :return: data loaders
        """
        cpus = cpu_count()
        train_tforms, eval_tforms = self.create_transformer()

        if 'digits' in self.args.cfg:
            trs_set, trt_set, tes_set, tet_set = self.create_digits_datasets(train_tforms, eval_tforms)
        elif 'office' in self.args.cfg:
            trs_set, trt_set, tes_set, tet_set = self.create_office_datasets(train_tforms, eval_tforms)
        elif 'visda' in self.args.cfg:
            trs_set, trt_set, tes_set, tet_set = self.create_visda_datasets(train_tforms, eval_tforms)
        else:
            raise NotImplementedError

        self.train_src_loader = DataLoader(trs_set, self.args.bs, shuffle=True, num_workers=cpus)
        self.train_tgt_loader = DataLoader(trt_set, self.args.bs, shuffle=True, num_workers=cpus)
        self.test_src_loader = DataLoader(tes_set, self.args.bs, shuffle=False, num_workers=cpus)
        self.test_tgt_loader = DataLoader(tet_set, self.args.bs, shuffle=False, num_workers=cpus)

    def load_digits_cfg(self):
        cfg = load_json(self.args.cfg)
        cfg = split_digits_train_test(cfg, self.args.src.upper(), self.args.tgt.upper(), 1, self.args.seed)

        trs = cfg[self.args.src.upper()]['TR']
        trt = cfg[self.args.tgt.upper()]['TR']
        tes = cfg[self.args.src.upper()]['TE']
        tet = cfg[self.args.tgt.upper()]['TE']

        return trs, trt, tes, tet

    def load_office_cfg(self):
        cfg = load_json(self.args.cfg)
        cfg = split_office_train_test(cfg, 1, self.args.seed)

        trs = cfg[self.args.src.upper()]['SRC-TR']
        trt = cfg[self.args.tgt.upper()]['TGT-TR']
        tes = cfg[self.args.src.upper()]['TGT-TE']
        tet = cfg[self.args.tgt.upper()]['TGT-TE']

        return trs, trt, tes, tet

    def load_visda_cfg(self):
        cfg = load_json(self.args.cfg)

        trs = cfg['SRC']['TRAIN']
        trt = cfg['TGT']['TRAIN']
        tes = cfg['SRC']['TRAIN']
        tet = cfg['TGT']['TEST']
        self.label_dict = cfg['Label']

        return trs, trt, tes, tet

    def create_digits_datasets(self, train_tforms, eval_tforms):
        """
        Create digits datasets
        :param train_tforms: training transformers
        :param eval_tforms: evaluation transformers
        :return:
            trs_set: training source set
            trt_set: training target set
            tes_set: testing source set
            tet_set: testing target set
        """
        trs, trt, tes, tet = self.load_digits_cfg()

        if self.args.aug_tgt_only:
            trs_set = DomainArrayDataset(trs, tforms=train_tforms)
        else:
            trs_set = DomainArrayDataset(trs, tforms=eval_tforms)
        trt_set = DomainArrayDataset(trt, tforms=train_tforms)
        tes_set = DomainArrayDataset(tes, tforms=eval_tforms)
        tet_set = DomainArrayDataset(tet, tforms=eval_tforms)

        return trs_set, trt_set, tes_set, tet_set

    def create_office_datasets(self, train_tforms, eval_tforms):
        """
        Create Office datasets
        :param train_tforms: training transformers
        :param eval_tforms: evaluation transformers
        :return:
            trs_set: training source set
            trt_set: training target set
            tes_set: testing source set
            tet_set: testing target set
        """
        trs, trt, tes, tet = self.load_office_cfg()

        if self.args.aug_tgt_only:
            trs_set = DomainArrayDataset(trs, tforms=train_tforms)
        else:
            trs_set = DomainArrayDataset(trs, tforms=eval_tforms)
        trt_set = DomainFolderDataset(trt, tforms=train_tforms)
        tes_set = DomainFolderDataset(tes, tforms=eval_tforms)
        tet_set = DomainFolderDataset(tet, tforms=eval_tforms)

        return trs_set, trt_set, tes_set, tet_set

    def create_visda_datasets(self, train_tforms, eval_tforms):
        """
        Create VisDA17 datasets
        :param train_tforms: training transformers
        :param eval_tforms: evaluation transformers
        :return:
            trs_set: training source set
            trt_set: training target set
            tes_set: testing source set
            tet_set: testing target set
        """
        # Read config
        trs, trt, tes, tet = self.load_visda_cfg()

        if self.args.aug_tgt_only:
            trs_set = DomainArrayDataset(trs, tforms=train_tforms)
        else:
            trs_set = DomainArrayDataset(trs, tforms=eval_tforms)
        trt_set = DomainRecDataset(trt, tforms=train_tforms)
        tes_set = DomainRecDataset(tes, tforms=eval_tforms)
        tet_set = DomainRecDataset(tet, tforms=eval_tforms)

        return trs_set, trt_set, tes_set, tet_set

    @staticmethod
    def create_metrics():
        """
        Create metrics
        :return: metrics
        """
        metrics = {'Train-Xent-Src': Loss(),
                   'Train-Xent-Tgt': Loss(),
                   'Train-Acc-Src': Accuracy(),
                   'Train-Acc-Tgt': Accuracy(),
                   'Train-Aux-Src': Loss(),
                   'Train-Aux-Tgt': Loss(),
                   'Train-Total-Src': Loss(),
                   'Train-Total-Tgt': Loss()}
        return metrics

    def create_lr_scheduler(self):
        """
        Create learning rate scheduler
        :return:
        """
        if len(self.args.lr_epochs) > 0 and self.args.optim is not 'adam':
            epochs = [int(e) for e in self.args.lr_epochs.split(',')]
            self.lr_schlr = MultiEpochScheduler(epochs, self.args.lr_factor)

    def reset_metrics(self):
        """
        Reset metrics
        :return:
        """
        for k, v in self.metrics.items():
            v.reset()

    def reset_record(self):
        """
        Reset training records
        :return:
        """
        self.records = {'Epoch': {'Epoch': 0, 'Src-Acc': 0., 'Src-preds': [], 'Src-label': [], 'Src-features': [],
                                  'Tgt-Acc': 0., 'Tgt-preds': [], 'Tgt-label': [], 'Tgt-features': []},
                        'Iter': {'Iter': 0, 'Src-Acc': 0., 'Src-preds': [], 'Src-label': [], 'Src-features': [],
                                 'Tgt-Acc': 0., 'Tgt-preds': [], 'Tgt-label': [], 'Tgt-features': []}}

    def create_trainer(self, inference):
        """
        Create trainer
        :param inference: network
        :return: trainer
        """

        if self.args.optim == 'sgd':
            optim_params = {'learning_rate': self.args.lr, 'wd': self.args.wd, 'momentum': self.args.mom}
        elif self.args.optim == 'adam':
            optim_params = {'learning_rate': self.args.lr, 'wd': self.args.wd}
        else:
            raise NotImplementedError

        trainer = Trainer(inference.collect_params(), optimizer=self.args.optim,
                          optimizer_params=optim_params)
        return trainer

    def create_inference(self):
        """
        Create inference model
        :return: network
        """
        if self.args.bb == 'lenetplus' or self.args.bb == 'conv2':
            inference = importlib.import_module('models.%s' % self.args.bb).get_inference(
                classes=self.args.nc, embed_size=self.args.embed_size, use_dropout=self.args.dropout,
                use_norm=self.args.l2n, use_angular=self.args.angular, use_inn=self.args.inn)
        elif self.args.bb == 'resnet':
            inference = importlib.import_module('models.%s' % self.args.bb).get_inference(
                num_layers=self.args.nlayers, classes=self.args.nc, embed_size=self.args.embed_size,
                use_dropout=self.args.dropout, thumbnail=False, use_norm=self.args.l2n,
                use_angular=self.args.angular)
        elif self.args.bb == 'vgg':
            inference = importlib.import_module('models.%s' % self.args.bb).get_inference(
                num_layers=self.args.nlayers, classes=self.args.nc, embed_size=self.args.embed_size,
                use_dropout=self.args.dropout, use_bn=self.args.bn, use_norm=self.args.l2n,
                use_angular=self.args.angular)
        else:
            raise NotImplementedError
        if self.args.hybridize:
            inference.hybridize()
        self.load_params(inference)

        return inference

    def train(self):
        """
        Training, need to be implemented after inheriting
        :return:
        """
        raise NotImplementedError

    def train_epoch(self, inference, trainer, **kwargs):
        """
        Training process for one epoch
        :param inference: inference
        :param trainer: trainer
        :param kwargs: additional arguments
        :return:
        """
        raise NotImplementedError

    def update_lr(self, trainer):
        """
        Update learning rate
        :param trainer: current trainer
        :return:
        """
        if self.lr_schlr is not None:
            if self.lr_schlr.update_lr(trainer, self.cur_epoch):
                self.logger.update_scalar('learning rate', trainer.learning_rate)

    def test(self):
        """
        Testing code
        """
        # create inference
        inference = self.create_inference()
        accs, ys, ys_hat, fs = self.eval(inference, self.test_src_loader, log=False)
        acct, yt, yt_hat, ft = self.eval(inference, self.test_tgt_loader, log=False)
        print('Source Acc: %f %%, Target Acc: %f %%' % (float(accs) * 100, float(acct) * 100))
        if self.args.plot:
            self.plot_tsne(fs, ys, ft, yt)

        if self.args.save_preds:
            print('save predictions...')
            out_prefix = os.path.splitext(self.args.model_path)[0]
            np.savetxt('%s-ys.txt' % out_prefix, ys)
            np.savetxt('%s-yt.txt' % out_prefix, yt)
            np.savetxt('%s-ys-hat.txt' % out_prefix, ys_hat)
            np.savetxt('%s-yt-hat.txt' % out_prefix, yt_hat)
            np.savetxt('%s-fs.txt' % out_prefix, fs)
            np.savetxt('%s-ft.txt' % out_prefix, ft)

    def log_iter(self):
        """
        Log for every iteration
        :return:
        """
        raise NotImplementedError

    def log_epoch(self):
        """
        Log for every epoch
        :return:
        """
        raise NotImplementedError

    def eval(self, inference, val_loader, log=True, target=True, epoch=True):
        """
        Evaluate the model
        :param inference: network
        :param val_loader: data loader
        :param log: log flag
        :param target: target flag for updating the record and log
        :param epoch: epoch flag for updating the record and log
        :return:
        """
        mtc_acc = Accuracy()
        mtc_acc.reset()
        # val_loader.reset()

        feature_nest, y_nest, y_hat_nest = [], [], []
        for X, Y in val_loader:
            X_lst = split_and_load(X, self.args.ctx, even_split=False)
            Y_lst = split_and_load(Y, self.args.ctx, even_split=False)

            for x, y in zip(X_lst, Y_lst):
                y_hat, features = inference(x)
                # update metric
                mtc_acc.update([y], [y_hat])

                y_nest.extend(y.asnumpy())
                feature_nest.extend(features.asnumpy())
                y_hat_nest.extend(y_hat.asnumpy())

        feature_nest = np.array(feature_nest)
        y_nest = np.array(y_nest).astype(int)
        y_hat_nest = np.array(y_hat_nest)

        if log:
            target_key = 'Tgt' if target else 'Src'
            epoch_key = 'Epoch' if epoch else 'Iter'
            record = self.cur_epoch if epoch else self.cur_iter

            if mtc_acc.get()[1] > self.records[epoch_key]['%s-Acc' % target_key]:
                if target:
                    self.records[epoch_key][epoch_key] = record
                self.records[epoch_key]['%s-Acc' % target_key] = mtc_acc.get()[1]
                self.records[epoch_key]['%s-label' % target_key] = y_nest
                self.records[epoch_key]['%s-preds' % target_key] = y_hat_nest
                self.records[epoch_key]['%s-features' % target_key] = feature_nest

                self.save_params(inference, 0, epoch_key)

            self.logger.update_scalar('%s [%d]: Eval-Acc-%s' % (epoch_key, record, target_key), mtc_acc.get()[1])
            if self.sw:
                self.sw.add_scalar('Acc/Eval-%s-Acc-%s' % (epoch, target_key), mtc_acc.get()[1], global_step=record)

        return mtc_acc.get()[1], y_nest, y_hat_nest, feature_nest

    def load_params(self, inference, init=initializer.Uniform(), postfix='epoch'):
        """
        load the parameters
        :param inference: network
        :param init: initializer function
        :param postfix: postfix
        :return:
        """
        if self.args.training:
            if self.args.pretrained:
                # print('load the weights from path: %s' % self.args.model_path)
                print('load the weights for features from path: %s' % self.args.model_path)
                inference.features.load_parameters(self.args.model_path, self.args.ctx, ignore_extra=True)
                print('initialize the weights for embeds and output')
                inference.embeds.initialize(init=initializer.Xavier(magnitude=2.24), ctx=self.args.ctx)
                inference.output.initialize(init=initializer.Xavier(magnitude=2.24), ctx=self.args.ctx)
            elif self.args.model_path.endswith('.params'):
                print('load the weights from path: %s' % self.args.model_path)
                inference.load_parameters(self.args.model_path, self.args.ctx)
            elif self.args.start_epoch > 0:
                print('load the weights from path: %s' % os.path.join(self.args.ckpt, '%s-%s-%04d.params'
                                                                      % (self.args.bb, postfix, 0)))
                inference.load_parameters(os.path.join(self.args.ckpt, '%s-%s-%04d.params' %
                                          (self.args.bb, postfix, 0)), self.args.ctx)
            else:
                print('Initialize the weights')
                inference.initialize(init, ctx=self.args.ctx)
        else:
            print('load the weights from path: %s' % self.args.model_path)
            inference.load_parameters(self.args.model_path, self.args.ctx)

    def save_params(self, inference, epoch=0, postfix='epoch'):
        """
        Save the model
        :param inference: network
        :param epoch: current epoch
        :param postfix: name
        :return:
        """
        if postfix.lower() == 'epoch':
            inference.save_parameters(os.path.join(self.args.ckpt, '%s-%s-%04d.params' % (self.args.bb, 'epoch', epoch)))
        else:
            inference.save_parameters(os.path.join(self.args.ckpt, '%s-%s-%08d.params' % (self.args.bb, 'iter', epoch)))

    def plot_tsne(self, xs, ys, xt=None, yt=None):
        """
        Plot results using tSNE algorithm
        :param xs: source features
        :param ys: source labels
        :param xt: target features
        :param yt: target labels
        :param name_dicts: target labels
        :return:
        """
        save_path = os.path.join(self.args.log, '%s-%s' % (self.args.method, self.args.postfix) + '.pdf')

        print('Saving image to %s' % save_path)
        if xt is None or yt is None:
            cal_tsne_embeds(xs, ys, save_path=save_path)
        else:
            cal_tsne_embeds_src_tgt(xs, ys, xt, yt, save_path=save_path, names=self.label_dict)


class ClsModel(DomainModel):
    """
    Basic Classification Model for one domain
    """
    def __init__(self, args, train_tgt=True):
        super(ClsModel, self).__init__(args)
        self._train_tgt = train_tgt

    def train(self):
        """
        Training function
        :return: None
        """
        if self.args.training:
            # create inference
            inference = self.create_inference()
            # create trainer
            trainer = self.create_trainer(inference)

            for cur_epoch in range(self.args.start_epoch + 1, self.args.end_epoch + 1):
                self.cur_epoch = cur_epoch
                # reset metrics
                self.reset_metrics()
                # training
                self.train_epoch(inference, trainer)
                # update learning rate
                self.update_lr(trainer)

            self.logger.update_scalar('Epoch [%d]: Best-Acc=' % self.records['Epoch']['Epoch'],
                                      self.records['Epoch']['Tgt-Acc'])
            self.logger.update_scalar('Iter [%d]: Best-Acc=' % self.records['Iter']['Iter'],
                                      self.records['Iter']['Tgt-Acc'])

            if self.sw:
                self.sw.close()

    def train_epoch(self, inference, trainer, **kwargs):
        """
        Train on the both source and target domain
        :param inference:
        :param trainer:
        :return:
        """
        criterion_xent = SoftmaxCrossEntropyLoss()

        if self.args.train_src:
            for xs, ys in self.train_src_loader:
                xs_lst = split_and_load(xs, self.args.ctx, even_split=False)
                ys_lst = split_and_load(ys, self.args.ctx, even_split=False)
                with autograd.record():
                    loss = []
                    for x, y in zip(xs_lst, ys_lst):
                        y_hat, _ = inference(x)
                        loss_xent = criterion_xent(y_hat, y)
                        loss.append(loss_xent)
                        self.metrics['Train-Xent-Src'].update(None, [loss_xent])
                        self.metrics['Train-Acc-Src'].update([y], [y_hat])

                        self.cur_iter += 1

                    for l in loss:
                        l.backward()

                trainer.step(xs.shape[0])

                self.log_src = True
                if self.args.log_itv > 0 and self.cur_iter % self.args.log_itv == 0:
                    # evaluate
                    self.log_iter()
                    if self.args.eval:
                        self.eval(inference, self.test_tgt_loader, target=False, epoch=False)
                    # reset to False
                    self.log_src = False

        if self._train_tgt:
            for xt, yt in self.train_tgt_loader:
                xt_lst = split_and_load(xt, self.args.ctx, even_split=False)
                yt_lst = split_and_load(yt, self.args.ctx, even_split=False)

                # compute the loss
                with autograd.record():
                    loss = []
                    for x, y in zip(xt_lst, yt_lst):
                        y_hat, _ = inference(x)
                        loss_xent = criterion_xent(y_hat, y)
                        loss.append(loss_xent)

                        self.metrics['Train-Xent-Tgt'].update(None, [loss_xent])
                        self.metrics['Train-Acc-Tgt'].update([y], [y_hat])

                        self.cur_iter += 1

                    for l in loss:
                        l.backward()

                trainer.step(xt.shape[0])

                self.log_tgt = True

                if self.args.log_itv > 0 and self.cur_iter % self.args.log_itv == 0:
                    self.log_iter()
                    if self.args.eval:
                        self.eval(inference, self.test_tgt_loader, target=True, epoch=False)
                    # reset to False in case log in the training with source
                    self.log_tgt = False

        self.log_epoch()
        if self.args.eval and self.cur_epoch > self.args.eval_epoch:
            self.eval(inference, self.test_tgt_loader, target=True)

    def log_iter(self):
        if self.log_src:
            self.logger.update_scalar('Iter [%d]: Train-Xent-Src' % self.cur_iter,
                                      self.metrics['Train-Xent-Src'].get()[1])
            self.logger.update_scalar('Iter [%d]: Train-Acc-Src' % self.cur_iter,
                                      self.metrics['Train-Acc-Src'].get()[1])

        if self.log_tgt:
            self.logger.update_scalar('Iter [%d]: Train-Xent-Tgt' % self.cur_iter,
                                      self.metrics['Train-Xent-Tgt'].get()[1])
            self.logger.update_scalar('Iter [%d]: Train-Acc-Tgt' % self.cur_iter,
                                      self.metrics['Train-Acc-Tgt'].get()[1])
        if self.sw:
            if self.log_src:
                self.sw.add_scalar('Loss/Train-Iter-XEnt-Src', self.metrics['Train-Xent-Src'].get()[1],
                                   global_step=self.cur_iter)
                self.sw.add_scalar('Acc/Train-Iter-Acc-Src', self.metrics['Train-Acc-Src'].get()[1],
                                   global_step=self.cur_iter)
            if self.log_tgt:
                self.sw.add_scalar('Loss/Train-Iter-XEnt-Tgt', self.metrics['Train-Xent-Tgt'].get()[1],
                                   global_step=self.cur_iter)
                self.sw.add_scalar('Acc/Train-Iter-Acc-Tgt', self.metrics['Train-Acc-Tgt'].get()[1],
                                   global_step=self.cur_iter)

    def log_epoch(self):
        if self.log_src:
            self.logger.update_scalar('Epoch [%d]: Train-XEnt-Src' % self.cur_epoch,
                                      self.metrics['Train-Xent-Src'].get()[1])
            self.logger.update_scalar('Epoch [%d]: Train-Acc-Src' % self.cur_epoch,
                                      self.metrics['Train-Acc-Src'].get()[1])
        if self.log_tgt:
            self.logger.update_scalar('Epoch [%d]: Train-XEnt-Tgt' % self.cur_epoch,
                                      self.metrics['Train-Xent-Tgt'].get()[1])

            self.logger.update_scalar('Epoch [%d]: Train-Acc-Tgt' % self.cur_epoch,
                                      self.metrics['Train-Acc-Tgt'].get()[1])

        if self.sw:
            if not self.args.pretrained:
                self.sw.add_scalar('Loss/Train-Epoch-XEnt-Src', self.metrics['Train-Xent-Src'].get()[1],
                                   global_step=self.cur_epoch)
                self.sw.add_scalar('Acc/Train-Epoch-Acc-Src', self.metrics['Train-Acc-Src'].get()[1],
                                   global_step=self.cur_epoch)
            self.sw.add_scalar('Loss/Train-Epoch-XEnt-Tgt', self.metrics['Train-Xent-Tgt'].get()[1],
                               global_step=self.cur_epoch)
            self.sw.add_scalar('Acc/Train-Epoch-Acc-Tgt', self.metrics['Train-Acc-Tgt'].get()[1],
                               global_step=self.cur_epoch)


class AuxModel(DomainModel):
    def __init__(self, args):
        super(AuxModel, self).__init__(args)

    def create_loader(self):
        """
        Overwrite the data loader function
        :return: pairwised data loader, None, eval source loader, test target loader
        """
        cpus = cpu_count()

        train_tforms, eval_tforms = [transforms.Resize(self.args.resize)], [transforms.Resize(self.args.resize)]

        if self.args.random_crop:
            train_tforms.append(transforms.RandomResizedCrop(self.args.size, scale=(0.8, 1.2)))
        else:
            train_tforms.append(transforms.CenterCrop(self.args.size))

        eval_tforms.append(transforms.CenterCrop(self.args.size))

        if self.args.flip:
            train_tforms.append(transforms.RandomFlipLeftRight())

        if self.args.random_color:
            train_tforms.append(transforms.RandomColorJitter(self.args.color_jitter, self.args.color_jitter,
                                                             self.args.color_jitter, 0.1))

        train_tforms.extend([transforms.ToTensor(), transforms.Normalize(self.args.mean, self.args.std)])
        eval_tforms.extend([transforms.ToTensor(), transforms.Normalize(self.args.mean, self.args.std)])

        train_tforms = transforms.Compose(train_tforms)
        eval_tforms = transforms.Compose(eval_tforms)

        if 'digits' in self.args.cfg:
            trs_set, tes_set, tet_set = self.create_digits_datasets(train_tforms, eval_tforms)
        elif 'office' in self.args.cfg:
            trs_set, tes_set, tet_set = self.create_office_datasets(train_tforms, eval_tforms)
        elif 'visda' in self.args.cfg:
            trs_set, tes_set, tet_set = self.create_visda_datasets(train_tforms, eval_tforms)
        else:
            raise NotImplementedError

        self.train_src_loader = DataLoader(trs_set, self.args.bs, shuffle=True, num_workers=cpus)
        self.test_src_loader = DataLoader(tes_set, self.args.bs, shuffle=False, num_workers=cpus)
        self.test_tgt_loader = DataLoader(tet_set, self.args.bs, shuffle=False, num_workers=cpus)

    def create_digits_datasets(self, train_tforms, eval_tforms):
        """
        Create digits datasets
        :param train_tforms: training transformers
        :param eval_tforms: evaluation transformers
        :return:
            trs_set: training source set
            trt_set: training target set
            tes_set: testing source set
            tet_set: testing target set
        """
        cfg = load_json(self.args.cfg)
        cfg = split_digits_train_test(cfg, self.args.src.upper(), self.args.tgt.upper(), 1, self.args.seed)

        trs = cfg[self.args.src.upper()]['TR']
        trt = cfg[self.args.tgt.upper()]['TR']
        tes = cfg[self.args.src.upper()]['TE']
        tet = cfg[self.args.tgt.upper()]['TE']

        trs_set = DomainArrayDataset(trs, trt, tforms=train_tforms, tformt=train_tforms, ratio=self.args.ratio)
        tes_set = DomainArrayDataset(tes, tforms=eval_tforms)
        tet_set = DomainArrayDataset(tet, tforms=eval_tforms)

        return trs_set, tes_set, tet_set

    def create_office_datasets(self, train_tforms, eval_tforms):
        """
        Create office datasets
        :param train_tforms: training transformers
        :param eval_tforms: evaluation transformers
        :return:
            trs_set: training source set
            trt_set: training target set
            tes_set: testing source set
            tet_set: testing target set
        """
        cfg = load_json(self.args.cfg)
        cfg = split_office_train_test(cfg, 1, self.args.seed)

        trs = cfg[self.args.src.upper()]['SRC-TR']
        trt = cfg[self.args.tgt.upper()]['TGT-TR']
        tes = cfg[self.args.src.upper()]['TGT-TE']
        tet = cfg[self.args.tgt.upper()]['TGT-TE']

        trs_set = DomainFolderDataset(trs, trt, tforms=train_tforms, tformt=train_tforms, ratio=self.args.ratio)
        tes_set = DomainFolderDataset(tes, tforms=eval_tforms)
        tet_set = DomainFolderDataset(tet, tforms=eval_tforms)

        return trs_set, tes_set, tet_set

    def create_visda_datasets(self, train_tforms, eval_tforms):
        """
        Create visda datasets
        :param train_tforms: training transformers
        :param eval_tforms: evaluation transformers
        :return:
            trs_set: training source set
            trt_set: training target set
            tes_set: testing source set
            tet_set: testing target set
        """
        # Read config
        cfg = load_json(self.args.cfg)

        trs = cfg['SRC']['TRAIN']
        trt = cfg['TGT']['TRAIN']
        tes = cfg['SRC']['TRAIN']
        tet = cfg['TGT']['TEST']

        self.label_dict = cfg['Label']

        trs_set = DomainRecDataset(trs, trt, tforms=train_tforms, tformt=train_tforms, ratio=self.args.ratio)
        tes_set = DomainRecDataset(tes, tforms=eval_tforms)
        tet_set = DomainRecDataset(tet, tforms=eval_tforms)

        return trs_set, tes_set, tet_set

    def train(self):
        """
        Training process for Auxiliary model
        :return: None
        """
        # create inference
        inference = self.create_inference()
        # create trainer
        trainer = self.create_trainer(inference)

        for cur_epoch in range(self.args.start_epoch + 1, self.args.end_epoch + 1):
            self.cur_epoch = cur_epoch
            # reset metrics
            self.reset_metrics()
            # training
            self.train_epoch(inference, trainer)
            # update learning rate
            self.update_lr(trainer)

        self.logger.update_scalar('Epoch [%d]: Best-Acc=' % self.records['Epoch']['Epoch'],
                                  self.records['Epoch']['Tgt-Acc'])
        self.logger.update_scalar('Iter [%d]: Best-Acc=' % self.records['Iter']['Iter'],
                                  self.records['Iter']['Tgt-Acc'])

        if self.sw:
            self.sw.close()

    def train_epoch(self, inference, trainer, **kwargs):
        """
        Training process for one epoch
        :param inference: inference network
        :param trainer: trainer
        :return:
        """
        raise NotImplementedError

    def log_iter(self):
        if self.args.train_src:
            self.logger.update_scalar('Iter [%d]: Train-Xent-Src' % self.cur_iter,
                                      self.metrics['Train-Xent-Src'].get()[1])
            self.logger.update_scalar('Iter [%d]: Train-Aux-Src' % self.cur_iter,
                                      self.metrics['Train-Aux-Src'].get()[1])
            self.logger.update_scalar('Iter [%d]: Train-Total-Src' % self.cur_iter,
                                      self.metrics['Train-Total-Src'].get()[1])
            self.logger.update_scalar('Iter [%d]: Train-Acc-Src' % self.cur_iter,
                                      self.metrics['Train-Acc-Src'].get()[1])

        self.logger.update_scalar('Iter [%d]: Train-Xent-Tgt' % self.cur_iter,
                                  self.metrics['Train-Xent-Tgt'].get()[1])
        self.logger.update_scalar('Iter [%d]: Train-Aux-Tgt' % self.cur_iter,
                                  self.metrics['Train-Aux-Tgt'].get()[1])
        self.logger.update_scalar('Iter [%d]: Train-Total-Tgt' % self.cur_iter,
                                  self.metrics['Train-Total-Tgt'].get()[1])
        self.logger.update_scalar('Iter [%d]: Train-Acc-Tgt' % self.cur_iter,
                                  self.metrics['Train-Acc-Tgt'].get()[1])

        if self.sw:
            if self.args.train_src:
                self.sw.add_scalar('Loss/Train-Iter-XEnt-Src', self.metrics['Train-Xent-Src'].get()[1],
                                   global_step=self.cur_iter)
                self.sw.add_scalar('Loss/Train-Iter-Aux-Src', self.metrics['Train-Aux-Src'].get()[1],
                                   global_step=self.cur_iter)
                self.sw.add_scalar('Acc/Train-Iter-Total-Src', self.metrics['Train-Total-Src'].get()[1],
                                   global_step=self.cur_iter)
                self.sw.add_scalar('Acc/Train-Iter-Acc-Src', self.metrics['Train-Acc-Src'].get()[1],
                                   global_step=self.cur_iter)

            self.sw.add_scalar('Loss/Train-Iter-XEnt-Tgt', self.metrics['Train-Xent-Tgt'].get()[1],
                               global_step=self.cur_iter)
            self.sw.add_scalar('Loss/Train-Iter-Aux-Tgt', self.metrics['Train-Aux-Tgt'].get()[1],
                               global_step=self.cur_iter)
            self.sw.add_scalar('Acc/Train-Iter-Total-Tgt', self.metrics['Train-Total-Tgt'].get()[1],
                               global_step=self.cur_iter)
            self.sw.add_scalar('Acc/Train-Iter-Acc-Tgt', self.metrics['Train-Acc-Tgt'].get()[1],
                               global_step=self.cur_iter)

    def log_epoch(self):
        if self.args.train_src:
            self.logger.update_scalar('Epoch [%d]: Train-XEnt-Src' % self.cur_epoch,
                                      self.metrics['Train-Xent-Src'].get()[1])
            self.logger.update_scalar('Epoch [%d]: Train-Aux-Src' % self.cur_epoch,
                                      self.metrics['Train-Aux-Src'].get()[1])
            self.logger.update_scalar('Epoch [%d]: Train-Total-Src' % self.cur_epoch,
                                      self.metrics['Train-Total-Src'].get()[1])
            self.logger.update_scalar('Epoch [%d]: Train-Acc-Src' % self.cur_epoch,
                                      self.metrics['Train-Acc-Src'].get()[1])

        self.logger.update_scalar('Epoch [%d]: Train-XEnt-Tgt' % self.cur_epoch,
                                  self.metrics['Train-Xent-Tgt'].get()[1])
        self.logger.update_scalar('Epoch [%d]: Train-Aux-Tgt' % self.cur_epoch,
                                  self.metrics['Train-Aux-Tgt'].get()[1])
        self.logger.update_scalar('Epoch [%d]: Train-Total-Tgt' % self.cur_epoch,
                                  self.metrics['Train-Total-Tgt'].get()[1])
        self.logger.update_scalar('Epoch [%d]: Train-Acc-Tgt' % self.cur_epoch,
                                  self.metrics['Train-Acc-Src'].get()[1])

        if self.sw:
            if self.args.train_src:
                self.sw.add_scalar('Loss/Train-Epoch-XEnt-Src', self.metrics['Train-Xent-Src'].get()[1],
                                   global_step=self.cur_epoch)
                self.sw.add_scalar('Loss/Train-Epoch-Aux-Src', self.metrics['Train-Aux-Src'].get()[1],
                                   global_step=self.cur_epoch)
                self.sw.add_scalar('Acc/Train-Epoch-Total-Src', self.metrics['Train-Total-Src'].get()[1],
                                   global_step=self.cur_epoch)
                self.sw.add_scalar('Acc/Train-Epoch-Acc-Src', self.metrics['Train-Acc-Src'].get()[1],
                                   global_step=self.cur_epoch)

            self.sw.add_scalar('Loss/Train-Epoch-XEnt-Tgt', self.metrics['Train-Xent-Tgt'].get()[1],
                               global_step=self.cur_epoch)
            self.sw.add_scalar('Loss/Train-Epoch-Aux-Tgt', self.metrics['Train-Aux-Tgt'].get()[1],
                               global_step=self.cur_epoch)
            self.sw.add_scalar('Acc/Train-Epoch-Total-Tgt', self.metrics['Train-Total-Tgt'].get()[1],
                               global_step=self.cur_epoch)
            self.sw.add_scalar('Acc/Train-Epoch-Acc-Tgt', self.metrics['Train-Acc-Tgt'].get()[1],
                               global_step=self.cur_epoch)


class CCSA(AuxModel):
    def __init__(self, args):
        super(CCSA, self).__init__(args)

    def train_epoch(self, inference, trainer, **kwargs):
        criterion_xent = SoftmaxCrossEntropyLoss()

        criterion_aux = ContrastiveLoss()

        for Xs, Ys, Xt, Yt, Yc in self.train_src_loader:
            Xs_lst = split_and_load(Xs, self.args.ctx, even_split=False)
            Ys_lst = split_and_load(Ys, self.args.ctx, even_split=False)
            Xt_lst = split_and_load(Xt, self.args.ctx, even_split=False)
            Yt_lst = split_and_load(Yt, self.args.ctx, even_split=False)
            Yc_lst = split_and_load(Yc, self.args.ctx, even_split=False)

            with autograd.record():
                loss = []
                for xs, ys, xt, yt, yc in zip(Xs_lst, Ys_lst, Xt_lst, Yt_lst, Yc_lst):
                    ys_hat, fts = inference(xs)
                    yt_hat, ftt = inference(xt)

                    loss_xent_src = criterion_xent(ys_hat, ys)
                    loss_aux = criterion_aux(fts - ftt, yc)

                    loss_total = (1 - self.args.alpha) * loss_xent_src + self.args.alpha * loss_aux

                    loss.append(loss_total)

                    self.metrics['Train-Xent-Src'].update(None, [loss_xent_src])
                    self.metrics['Train-Acc-Src'].update([Ys], [ys_hat])
                    self.metrics['Train-Acc-Tgt'].update([Yt], [yt_hat])
                    self.metrics['Train-Aux-Src'].update(None, [loss_aux])
                    self.metrics['Train-Total-Src'].update(None, [loss_total])

                for l in loss:
                    l.backward()

            trainer.step(Xs.shape[0])

            with autograd.record():
                loss = []
                for xs, ys, xt, yt, yc in zip(Xs_lst, Ys_lst, Xt_lst, Yt_lst, Yc_lst):
                    yt_hat, ftt = inference(xt)
                    ys_hat, fts = inference(xs)

                    loss_xent_tgt = criterion_xent(yt_hat, yt)
                    loss_aux = criterion_aux(fts - ftt, yc)

                    loss_total = (1 - self.args.alpha) * loss_xent_tgt + self.args.alpha * loss_aux

                    loss.append(loss_total)

                    self.metrics['Train-Xent-Tgt'].update(None, [loss_xent_tgt])
                    self.metrics['Train-Acc-Src'].update([Ys], [ys_hat])
                    self.metrics['Train-Acc-Tgt'].update([Yt], [yt_hat])

                    self.metrics['Train-Aux-Tgt'].update(None, [loss_aux])
                    self.metrics['Train-Total-Tgt'].update(None, [loss_total])

                for l in loss:
                    l.backward()

            trainer.step(Xt.shape[0])

            self.cur_iter += 1

            if self.args.log_itv > 0 and self.cur_iter % self.args.log_itv == 0:
                self.log_iter()
                if self.args.eval:
                    self.eval(inference, self.test_tgt_loader, target=True, epoch=False)

        self.log_epoch()
        if self.args.eval and self.cur_epoch > self.args.eval_epoch:
            self.eval(inference, self.test_tgt_loader, target=True, epoch=True)


class dSNE(AuxModel):
    def __init__(self, args):
        super(dSNE, self).__init__(args)

    def train_epoch(self, inference, trainer, **kwargs):
        """
        Training with dSNEt loss
        :param inference: inference
        :param trainer: trainer of inference
        :return:
        """

        for Xs, Ys, Xt, Yt, _ in self.train_src_loader:
            Xs_lst = split_and_load(Xs, self.args.ctx, even_split=False)
            Ys_lst = split_and_load(Ys, self.args.ctx, even_split=False)
            Xt_lst = split_and_load(Xt, self.args.ctx, even_split=False)
            Yt_lst = split_and_load(Yt, self.args.ctx, even_split=False)

            if self.args.train_src:
                self.train_batch(Xs_lst, Ys_lst, Xt_lst, Yt_lst, inference, target=False)
                trainer.step(Xs.shape[0])

            self.train_batch(Xt_lst, Yt_lst, Xs_lst, Ys_lst, inference, target=True)

            trainer.step(Xt.shape[0])

            if self.args.log_itv > 0 and self.cur_iter % self.args.log_itv == 0:
                self.log_iter()
                if self.args.eval:
                    self.eval(inference, self.test_tgt_loader, target=True, epoch=False)

        self.log_epoch()
        if self.args.eval and self.cur_epoch > self.args.eval_epoch:
            self.eval(inference, self.test_tgt_loader, target=True, epoch=True)

    def train_batch(self, Xs_lst, Ys_lst, Xt_lst, Yt_lst, inference, target=True):
        criterion_xent = SoftmaxCrossEntropyLoss()

        postfix = 'Tgt' if target else 'Src'

        with autograd.record():
            loss = []
            for xs, ys, xt, yt in zip(Xs_lst, Ys_lst, Xt_lst, Yt_lst):
                criterion_aux = dSNELoss(xs.shape[0], xt.shape[0], self.args.embed_size, self.args.margin,
                                         self.args.fn)

                ys_hat, fts = inference(xs)
                yt_hat, ftt = inference(xt)

                loss_xent_src = criterion_xent(ys_hat, ys)
                loss_aux = criterion_aux(fts, ys, ftt, yt)

                loss_total = (1 - self.args.alpha) * loss_xent_src + self.args.alpha * loss_aux
                loss.append(loss_total)

                self.metrics['Train-Xent-%s' % postfix].update(None, [loss_xent_src])
                self.metrics['Train-Acc-Src'].update([ys], [ys_hat])
                self.metrics['Train-Acc-Tgt'].update([yt], [yt_hat])
                self.metrics['Train-Aux-%s' % postfix].update(None, [loss_aux])
                self.metrics['Train-Total-%s' % postfix].update(None, [loss_total])

                self.cur_iter += 1

            for l in loss:
                l.backward()
