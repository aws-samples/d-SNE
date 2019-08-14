from mxnet.metric import Accuracy
from mxnet import autograd
import numpy as np


def eval_acc(inference, val_loader, ctx, return_meta=False):
    mtc_acc = Accuracy()
    mtc_acc.reset()

    feature_nest, y_nest, y_hat_nest = [], [], []
    for X, y in val_loader:
        X = X.as_in_context(ctx[0])
        y = y.as_in_context(ctx[0])
        with autograd.record(train_mode=False):
            y_hat, features = inference(X)

        # update metric
        mtc_acc.update([y], [y_hat])

        if return_meta:
            y_nest.extend(y.asnumpy())
            feature_nest.extend(features.asnumpy())
            y_hat_nest.extend(y_hat.asnumpy())

    feature_nest = np.array(feature_nest)
    y_nest = np.array(y_nest)
    y_hat_nest = np.array(y_hat_nest)

    if return_meta:
        return mtc_acc.get()[1], y_nest, y_hat_nest, feature_nest

    return mtc_acc.get()[1]

