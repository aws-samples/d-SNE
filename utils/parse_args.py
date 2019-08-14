"""
Argument parser
"""
import mxnet as mx
import argparse
import os


def parse_args_sda():

    parser = argparse.ArgumentParser(description='Supervised Domain Adaptation for main function')
    # os
    os_parser = parser.add_argument_group(description='os related argument parser')
    os_parser.add_argument('--ckpt', type=str, default='ckpt', help='checkpoint directory')
    os_parser.add_argument('--log', type=str, default='log', help='log directory')
    os_parser.add_argument('--gpus', type=str, default='0', help='gpus, split by comma')
    os_parser.add_argument('--mxboard', action='store_true', help='use mxboard to log')
    os_parser.add_argument('--postfix', type=str, default='0', help='log/ckpt postfix')
    os_parser.add_argument('--log-itv', type=int, default=100, help='log batch interval')
    # training
    train_parser = parser.add_argument_group(description='train related argument parser')
    train_parser.add_argument('--method', type=str, default='v0',
                              help='methods: '
                                   'v0 -> train src, test tgt;'
                                   'v1 -> train src and tgt; '
                                   'ccsa -> contrastive loss; '
                                   'dsnet -> triplet loss')
    train_parser.add_argument('--cfg', type=str, default='cfg/visda17.json', help='experiments config')
    train_parser.add_argument('--src', type=str, default='S', help='source domain')
    train_parser.add_argument('--tgt', type=str, default='R', help='target domain')
    train_parser.add_argument('--nc', type=int, default=12, help='number of classes')
    train_parser.add_argument('--size', type=int, default=224, help='input shape')
    train_parser.add_argument('--bb', type=str, default='resnet', help='backbone network for feature extraction')
    train_parser.add_argument('--nlayers', type=int, default=101, help='number layers')
    train_parser.add_argument('--dropout', action='store_true', help='use dropout')
    train_parser.add_argument('--inn', action='store_true', help='use instance norm')
    train_parser.add_argument('--bn', action='store_true', help='use batch norm')
    train_parser.add_argument('--embed-size', type=int, default=512, help='embedding size')
    train_parser.add_argument('--pretrained', action='store_true', help='load pretrained')
    train_parser.add_argument('--model-path', type=str, default='', help='relative path in the ckpt folder')
    train_parser.add_argument('--start-epoch', type=int, default=0, help='start epoch')
    train_parser.add_argument('--end-epoch', type=int, default=20, help='total epochs')
    train_parser.add_argument('--seed', type=int, default=0, help='random seed')
    train_parser.add_argument('--train-src', action='store_true', help='train source images')
    train_parser.add_argument('--hybridize', action='store_true', help='hybridize the network')
    train_parser.add_argument('--best-value', type=float, default=0, help='best value')
    train_parser.add_argument('--alpha', type=float, default=0.1, help='alpha in the loss')
    train_parser.add_argument('--train', dest='training', action='store_true', help='training mode')
    train_parser.add_argument('--plot', action='store_true', help='plot the embedding features')
    train_parser.add_argument('--l2n', action='store_true', help='l2 normalize features')
    train_parser.add_argument('--fn', action='store_true', help='feature l2 normalization for the dSNEt')
    train_parser.add_argument('--angular', action='store_true', help='angularization')
    train_parser.add_argument('--aug-tgt-only', action='store_true', help='augment the target sample only')
    # evaluation
    eval_parser = parser.add_argument_group(description='evaluate arguments parser')
    eval_parser.add_argument('--no-eval', dest='eval', action='store_false', help='do not do evaluation')
    eval_parser.add_argument('--eval-epoch', type=int, default=0, help='the epoch will start to evaluate the model')
    # data loader
    loader_parser = parser.add_argument_group(description='data loader parser')
    loader_parser.add_argument('--bs', type=int, default=32, help='batch size for source domain')
    loader_parser.add_argument('--resize', type=int, default=224, help='resize image')
    loader_parser.add_argument('--mean', type=float, default=0.5, help='mean')
    loader_parser.add_argument('--std', type=float, default=0.5, help='std')
    loader_parser.add_argument('--color-jitter', type=float, default=0.2, help='color value')
    loader_parser.add_argument('--flip', action='store_true', help='random flip the data')
    loader_parser.add_argument('--random-crop', action='store_true', help='random crop')
    loader_parser.add_argument('--random-color', action='store_true', help='random color')
    loader_parser.add_argument('--ratio', type=int, default=1, help='negative/positive pairs')
    # optimizer
    optim_parser = parser.add_argument_group(description='optimizer parser')
    optim_parser.add_argument('--optim', type=str, default='sgd', help='optimizer')
    optim_parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    optim_parser.add_argument('--lr-epochs', type=str, default='', help='learning rate epochs')
    optim_parser.add_argument('--lr-factor', type=float, default=0.1, help='learning rate factor')
    optim_parser.add_argument('--wd', type=float, default=1e-4, help='weight decay')
    optim_parser.add_argument('--mom', type=float, default=0.9, help='momentum')
    # hyper-parameters
    hp_parser = parser.add_argument_group(description='hyper parameters')
    hp_parser.add_argument('--margin', type=float, default=1, help='margin')
    # testing
    test_parser = parser.add_argument_group(description='test related argument parser')
    test_parser.add_argument('--test', dest='training', action='store_false', help='testing mode')
    test_parser.add_argument('--save-preds', action='store_true', help='save predictions and features')

    parser.set_defaults(training=True, eval=True)
    args = parser.parse_args()

    args.ctx = [mx.cpu()] if args.gpus == '-1' else [mx.gpu(int(gpu_id)) for gpu_id in args.gpus.split(',')]
    args.log = os.path.join(args.log, 'S-%s_T-%s' % (args.src.upper(), args.tgt.upper()))
    args.ckpt = os.path.join(args.ckpt, 'S-%s_T-%s' % (args.src.upper(), args.tgt.upper()))

    args.model_path = os.path.join(args.ckpt, args.model_path)

    if args.bb == 'resnet' or args.bb == 'vgg':
        sub_dir = 'M-%s_bb-%s%d' % (args.method, args.bb, args.nlayers)
    else:
        sub_dir = 'M-%s_bb-%s' % (args.method, args.bb)

    args.ckpt = os.path.join(args.ckpt, sub_dir, '%s-%s' % (args.method, args.postfix))
    args.log = os.path.join(args.log, sub_dir, '%s-%s' % (args.method, args.postfix))

    if not os.path.exists(args.ckpt):
        os.makedirs(args.ckpt)

    return args


def parse_args_ssda():

    parser = argparse.ArgumentParser(description='Supervised Domain Adaptation for main function')
    # os
    os_parser = parser.add_argument_group(description='os related argument parser')
    os_parser.add_argument('--ckpt', type=str, default='ckpt', help='checkpoint directory')
    os_parser.add_argument('--log', type=str, default='log', help='log directory')
    os_parser.add_argument('--gpu', type=int, default=0, help='gpu')
    os_parser.add_argument('--mxboard', action='store_true', help='use mxboard to log')
    os_parser.add_argument('--postfix', type=str, default='0', help='log/ckpt postfix')
    os_parser.add_argument('--log-itv', type=int, default=0, help='log batch interval')
    # training
    train_parser = parser.add_argument_group(description='train related argument parser')
    train_parser.add_argument('--method', type=str, default='mtt', help='mean teacher model')
    train_parser.add_argument('--cfg', type=str, default='cfg/visda17.json', help='experiments config')
    train_parser.add_argument('--src', type=str, default='S', help='source domain')
    train_parser.add_argument('--tgt', type=str, default='R', help='target domain')
    train_parser.add_argument('--nc', type=int, default=12, help='number of classes')
    train_parser.add_argument('--size', type=int, default=224, help='input shape')
    train_parser.add_argument('--bb', type=str, default='resnet', help='backbone network for feature extraction')
    train_parser.add_argument('--nlayers', type=int, default=101, help='number layers')
    train_parser.add_argument('--dropout', action='store_true', help='use dropout')
    train_parser.add_argument('--inn', action='store_true', help='use instance norm')
    train_parser.add_argument('--bn', action='store_true', help='use batch norm')
    train_parser.add_argument('--embed-size', type=int, default=512, help='embedding size')
    train_parser.add_argument('--pretrained', action='store_true', help='load pretrained')
    train_parser.add_argument('--model-path', type=str, default='', help='relative path in the ckpt folder')
    train_parser.add_argument('--start-epoch', type=int, default=0, help='start epoch')
    train_parser.add_argument('--end-epoch', type=int, default=20, help='total epochs')
    train_parser.add_argument('--seed', type=int, default=0, help='random seed')
    train_parser.add_argument('--train-src', action='store_true', help='train source images')
    train_parser.add_argument('--hybridize', action='store_true', help='hybridize the network')
    train_parser.add_argument('--best-value', type=float, default=0, help='best value')
    train_parser.add_argument('--alpha', type=float, default=0.25, help='weight for auxiliary loss')
    train_parser.add_argument('--beta', type=float, default=1, help='weight for consistency loss')
    train_parser.add_argument('--train', dest='training', action='store_true', help='training mode')
    train_parser.add_argument('--plot', action='store_true', help='plot the embedding features')
    train_parser.add_argument('--l2n', action='store_true', help='l2 normalize features')
    train_parser.add_argument('--fn', action='store_true', help='feature l2 normalization for the dSNEt')
    train_parser.add_argument('--angular', action='store_true', help='angularization')
    train_parser.add_argument('--rampup-epoch', type=int, default=30, help='ramp up epoch')
    train_parser.add_argument('--thresh', type=float, default=0.9, help='mininum confidence threshold')
    # evaluation
    eval_parser = parser.add_argument_group(description='evaluate arguments parser')
    eval_parser.add_argument('--no-eval', dest='eval', action='store_false', help='do not do evaluation')
    eval_parser.add_argument('--eval-epoch', type=int, default=0, help='the epoch will start to evaluate the model')
    # data loader
    loader_parser = parser.add_argument_group(description='data loader parser')
    loader_parser.add_argument('--bs', type=int, default=32, help='batch size for source domain')
    loader_parser.add_argument('--resize', type=int, default=256, help='resize image')
    loader_parser.add_argument('--mean', type=float, default=0.5, help='mean')
    loader_parser.add_argument('--std', type=float, default=0.5, help='std')
    loader_parser.add_argument('--color-jitter', type=float, default=0.4, help='color value')
    loader_parser.add_argument('--flip', action='store_true', help='random flip the data')
    loader_parser.add_argument('--random-crop', action='store_true', help='random crop')
    loader_parser.add_argument('--random-color', action='store_true', help='random color')
    loader_parser.add_argument('--ratio', type=int, default=1, help='negative/positive pairs')
    # optimizer
    optim_parser = parser.add_argument_group(description='optimizer parser')
    optim_parser.add_argument('--optim', type=str, default='sgd', help='optimizer')
    optim_parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    optim_parser.add_argument('--lr-epochs', type=str, default='', help='learning rate epochs')
    optim_parser.add_argument('--lr-factor', type=float, default=0.1, help='learning rate factor')
    optim_parser.add_argument('--wd', type=float, default=1e-4, help='weight decay')
    optim_parser.add_argument('--mom', type=float, default=0.9, help='momentum')
    # hyper-parameters
    hp_parser = parser.add_argument_group(description='hyper parameters')
    hp_parser.add_argument('--margin', type=float, default=1, help='margin')
    hp_parser.add_argument('--ema-decay', type=float, default=0.999, help='ema alpha')
    # testing
    test_parser = parser.add_argument_group(description='test related argument parser')
    test_parser.add_argument('--test', dest='training', action='store_false', help='testing mode')
    test_parser.add_argument('--save-preds', action='store_true', help='save predictions and features')

    parser.set_defaults(training=True, eval=True)
    args = parser.parse_args()

    args.ctx = [mx.cpu()] if args.gpu < 0 else [mx.gpu(args.gpu)]
    args.log = os.path.join(args.log, 'S-%s_T-%s' % (args.src.upper(), args.tgt.upper()))
    args.ckpt = os.path.join(args.ckpt, 'S-%s_T-%s' % (args.src.upper(), args.tgt.upper()))

    args.model_path = os.path.join(args.ckpt, args.model_path)

    if args.bb == 'resnet' or args.bb == 'vgg':
        sub_dir = 'M-%s_bb-%s%d' % (args.method, args.bb, args.nlayers)
    else:
        sub_dir = 'M-%s_bb-%s' % (args.method, args.bb)

    args.ckpt = os.path.join(args.ckpt, sub_dir, '%s-%s' % (args.method, args.postfix))
    args.log = os.path.join(args.log, sub_dir, '%s-%s' % (args.method, args.postfix))

    if not os.path.exists(args.ckpt):
        os.makedirs(args.ckpt)

    return args
