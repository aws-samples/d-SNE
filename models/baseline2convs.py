"""
2 Convolutional layers according to CCSA code
"""
from mxnet import gluon
from train_val.custom_layers import AngularLinear


class Baseline2Convs(gluon.nn.HybridBlock):
    """
    LeNetPlus model
    """
    def __init__(self, classes=10, feature_size=256, use_dropout=True, use_norm=False, use_bn=False, use_noise=False,
                 **kwargs):
        super(Baseline2Convs, self).__init__(**kwargs)
        with self.name_scope():
            self.use_dropout = use_dropout
            self.use_norm = use_norm
            self.use_bn = use_bn
            self.use_noise = use_noise

            self.features = gluon.nn.HybridSequential(prefix='')

            self.features.add(gluon.nn.Conv2D(6, 5, activation='relu'))
            self.features.add(gluon.nn.Dropout(0.5))
            self.features.add(gluon.nn.Conv2D(16, 5, activation='relu'))
            self.features.add(gluon.nn.MaxPool2D())
            self.features.add(gluon.nn.Dropout(0.5))

            self.features.add(gluon.nn.Dense(feature_size))
            if self.use_norm:
                self.output = AngularLinear(classes, in_uints=feature_size)
            else:
                self.output = gluon.nn.Dense(classes)

    def hybrid_forward(self, F, x, *args, **kwargs):
        features = self.features(x)
        if self.use_norm:
            features = F.L2Normalization(features, mode='instance')
        outputs = self.output(features)
        return outputs, features


def get_inference(classes=10, embed_size=256, use_dropout=True, use_norm=False, use_bn=False, use_noise=False,
                  params=None):
    return Baseline2Convs(classes, embed_size, use_dropout, use_norm, use_bn, use_noise, params=params)
