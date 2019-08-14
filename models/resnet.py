"""
ResNet
"""
from mxnet.gluon.model_zoo.vision import BasicBlockV1, BasicBlockV2, BottleneckV1, BottleneckV2
from mxnet.gluon import HybridBlock, nn
from train_val.custom_layers import AngularLinear, L2Normalization

resnet_spec = {18: ('basic_block', [2, 2, 2, 2], [64, 64, 128, 256, 512]),
               34: ('basic_block', [3, 4, 6, 3], [64, 64, 128, 256, 512]),
               50: ('bottle_neck', [3, 4, 6, 3], [64, 256, 512, 1024, 2048]),
               101: ('bottle_neck', [3, 4, 23, 3], [64, 256, 512, 1024, 2048]),
               152: ('bottle_neck', [3, 8, 36, 3], [64, 256, 512, 1024, 2048])}


def _conv3x3(channels, stride, in_channels):
    return nn.Conv2D(channels, kernel_size=3, strides=stride, padding=1,
                     use_bias=False, in_channels=in_channels)


class ResNetV2(HybridBlock):
    r"""ResNet V2 model from
    `"Identity Mappings in Deep Residual Networks"
    <https://arxiv.org/abs/1603.05027>`_ paper.

    Parameters
    ----------
    block : HybridBlock
        Class for the residual block. Options are BasicBlockV1, BottleneckV1.
    layers : list of int
        Numbers of layers in each block
    channels : list of int
        Numbers of channels in each block. Length should be one larger than layers list.
    classes : int, default 1000
        Number of classification classes.
    thumbnail : bool, default False
        Enable thumbnail.
    """
    def __init__(self, block, layers, channels, classes=1000, embed_size=512, thumbnail=False, use_dropout=False,
                 use_norm=False, use_angular=False, **kwargs):
        super(ResNetV2, self).__init__(**kwargs)
        assert len(layers) == len(channels) - 1
        with self.name_scope():
            self.use_norm = use_norm
            self.use_angular = use_angular

            self.features = nn.HybridSequential(prefix='')
            self.features.add(nn.BatchNorm(scale=False, center=False))
            if thumbnail:
                self.features.add(_conv3x3(channels[0], 1, 0))
            else:
                self.features.add(nn.Conv2D(channels[0], 7, 2, 3, use_bias=False))
                self.features.add(nn.BatchNorm())
                self.features.add(nn.Activation('relu'))
                self.features.add(nn.MaxPool2D(3, 2, 1))

            in_channels = channels[0]
            for i, num_layer in enumerate(layers):
                stride = 1 if i == 0 else 2
                self.features.add(self._make_layer(block, num_layer, channels[i+1],
                                                   stride, i+1, in_channels=in_channels))
                in_channels = channels[i+1]
            self.features.add(nn.BatchNorm())
            self.features.add(nn.Activation('relu'))
            self.features.add(nn.GlobalAvgPool2D())
            self.features.add(nn.Flatten())

            self.embeds = nn.HybridSequential(prefix='')
            self.embeds.add(nn.Dense(4096, activation='relu', weight_initializer='normal',
                                     bias_initializer='zeros'))
            if use_dropout:
                self.embeds.add(nn.Dropout(rate=0.5))
            self.embeds.add(nn.Dense(embed_size, activation='relu', weight_initializer='normal',
                                     bias_initializer='zeros'))
            if use_dropout:
                self.embeds.add(nn.Dropout(rate=0.5))

            if self.use_norm:
                self.embeds.add(L2Normalization(mode='instance'))

            if self.use_angular:
                self.output = AngularLinear(classes, in_uints=embed_size)
            else:
                self.output = nn.Dense(classes, in_units=embed_size)

    def _make_layer(self, block, layers, channels, stride, stage_index, in_channels=0):
        layer = nn.HybridSequential(prefix='stage%d_' % stage_index)
        with layer.name_scope():
            layer.add(block(channels, stride, channels != in_channels, in_channels=in_channels,
                            prefix=''))
            for _ in range(layers-1):
                layer.add(block(channels, 1, False, in_channels=channels, prefix=''))
        return layer

    def hybrid_forward(self, F, x):
        features = self.features(x)
        embeds = self.embeds(features)
        y_hat = self.output(embeds)
        return y_hat, embeds


resnet_block_versions = [{'basic_block': BasicBlockV1, 'bottle_neck': BottleneckV1},
                         {'basic_block': BasicBlockV2, 'bottle_neck': BottleneckV2}]


def get_inference(num_layers=101, classes=31, embed_size=512, version=2, thumbnail=True, use_dropout=False,
                  use_norm=False, **kwargs):
    assert num_layers in resnet_spec, \
        "Invalid number of layers: %d. Options are %s" % (
            num_layers, str(resnet_spec.keys()))
    block_type, layers, channels = resnet_spec[num_layers]
    resnet_class = ResNetV2
    block_class = resnet_block_versions[version - 1][block_type]
    net = resnet_class(block_class, layers, channels, classes=classes, embed_size=embed_size, thumbnail=thumbnail,
                       use_dropout=use_dropout, use_norm=use_norm)

    return net
