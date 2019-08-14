"""
LeNetPlus changed from LeNet
"""
from mxnet import gluon
from train_val.custom_layers import AngularLinear, L2Normalization


def _make_conv_block(block_index, num_chan=32, num_layer=2, stride=1, pad=2):
    out = gluon.nn.HybridSequential(prefix='block_%d_' % block_index)
    with out.name_scope():
        for _ in range(num_layer):
            out.add(gluon.nn.Conv2D(num_chan, kernel_size=3, strides=stride, padding=pad))
            out.add(gluon.nn.LeakyReLU(alpha=0.2))
        out.add(gluon.nn.MaxPool2D())

    return out


class LeNetPlus(gluon.nn.HybridBlock):
    """
    LeNetPlus model
    """
    def __init__(self, classes=10, feature_size=256, use_dropout=True, use_norm=False, use_bn=False, use_inn=False,
                 use_angular=False, **kwargs):
        super(LeNetPlus, self).__init__(**kwargs)
        num_chans = [32, 64, 128]
        with self.name_scope():
            self.use_dropout = use_dropout
            self.use_norm = use_norm
            self.use_bn = use_bn
            self.use_inn = use_inn
            self.use_angular = use_angular

            self.features = gluon.nn.HybridSequential(prefix='')

            if self.use_inn:
                self.features.add(gluon.nn.InstanceNorm())

            for i, num_chan in enumerate(num_chans):
                if use_bn:
                    self.features.add(gluon.nn.BatchNorm())

                self.features.add(_make_conv_block(i, num_chan=num_chan))

                if use_dropout and i > 0:
                    self.features.add(gluon.nn.Dropout(0.5))

            self.features.add(gluon.nn.Dense(feature_size))

            if self.use_norm:
                self.features.add(L2Normalization(mode='instance'))

            if use_angular:
                self.output = AngularLinear(classes, in_uints=feature_size)
            else:
                self.output = gluon.nn.Dense(classes)

    def hybrid_forward(self, F, x, *args, **kwargs):
        features = self.features(x)
        outputs = self.output(features)
        return outputs, features


def get_inference(classes=10, embed_size=256, use_dropout=True, use_norm=False, use_bn=False, use_inn=False, **kwargs):
    """
    get inference
    :param classes: number of classes
    :param embed_size: embedding size
    :param use_dropout: flag for using dropout
    :param use_norm: flag for using feature normalization
    :param use_bn: flag for using batch normalization
    :param use_inn: flag for using instance normaliation
    :param kwargs: other key words
    :return:
    """
    return LeNetPlus(classes, embed_size, use_dropout, use_norm, use_bn, use_inn, **kwargs)
