"""
VGG Net
"""
from mxnet.gluon import HybridBlock, nn
from mxnet.initializer import Xavier

from train_val.custom_layers import AngularLinear, L2Normalization


class VGG(HybridBlock):
    r"""VGG model from the `"Very Deep Convolutional Networks for Large-Scale Image Recognition"
    <https://arxiv.org/abs/1409.1556>`_ paper.

    Parameters
    ----------
    layers : list of int
        Numbers of layers in each feature block.
    filters : list of int
        Numbers of filters in each feature block. List length should match the layers.
    classes : int, default 1000
        Number of classification classes.
    batch_norm : bool, default False
        Use batch normalization.
    """
    def __init__(self, layers, filters, classes=1000, embed_size=512, use_droput=False, use_norm=False,
                 batch_norm=False, use_angular=False, **kwargs):
        super(VGG, self).__init__(**kwargs)
        assert len(layers) == len(filters)
        self.use_norm = use_norm
        self.use_angular = use_angular
        with self.name_scope():
            self.features = self._make_features(layers, filters, batch_norm)

            self.embeds = nn.HybridSequential(prefix='')
            if use_droput:
                self.embeds.add(nn.Dropout(rate=0.5))
            self.embeds.add(nn.Dense(4096, activation='relu', weight_initializer='normal', bias_initializer='zeros'))
            if use_droput:
                self.embeds.add(nn.Dropout(rate=0.5))
            self.embeds.add(nn.Dense(embed_size, activation='relu', weight_initializer='normal', bias_initializer='zeros'))
            if use_droput:
                self.embeds.add(nn.Dropout(rate=0.5))

            if self.use_norm:
                self.embeds.add(L2Normalization(mode='instance'))

            if self.use_angular:
                self.output = AngularLinear(classes, in_uints=embed_size)
            else:
                self.output = nn.Dense(classes)

    def _make_features(self, layers, filters, batch_norm):
        featurizer = nn.HybridSequential(prefix='')
        for i, num in enumerate(layers):
            for _ in range(num):
                featurizer.add(nn.Conv2D(filters[i], kernel_size=3, padding=1,
                                         weight_initializer=Xavier(rnd_type='gaussian',
                                                                   factor_type='out',
                                                                   magnitude=2),
                                         bias_initializer='zeros'))
                if batch_norm:
                    featurizer.add(nn.BatchNorm())
                featurizer.add(nn.Activation('relu'))
            featurizer.add(nn.MaxPool2D(strides=2))
        return featurizer

    def hybrid_forward(self, F, x):
        features = self.features(x)
        embeds = self.embeds(features)
        if self.use_norm:
            embeds = F.L2Normalization(embeds, mode='instance')

        x = self.output(embeds)
        return x, embeds


# Specification
vgg_spec = {11: ([1, 1, 2, 2, 2], [64, 128, 256, 512, 512]),
            13: ([2, 2, 2, 2, 2], [64, 128, 256, 512, 512]),
            16: ([2, 2, 3, 3, 3], [64, 128, 256, 512, 512]),
            19: ([2, 2, 4, 4, 4], [64, 128, 256, 512, 512])}


# Constructors
def get_inference(num_layers=16, classes=31, embed_size=512, use_dropout=False, use_norm=False, use_bn=False, **kwargs):
    r"""VGG model from the `"Very Deep Convolutional Networks for Large-Scale Image Recognition"
    <https://arxiv.org/abs/1409.1556>`_ paper.

    Parameters
    ----------
    num_layers : int
        Number of layers for the variant of densenet. Options are 11, 13, 16, 19.
    classes : number of classes
    embed_size: embedding size
    use_dropout: use dropout
    use_norm: use l2 normalization
    use_bn: use batch normalization
    """
    layers, filters = vgg_spec[num_layers]
    net = VGG(layers, filters, classes, embed_size, use_dropout, use_norm, use_bn, **kwargs)

    return net
