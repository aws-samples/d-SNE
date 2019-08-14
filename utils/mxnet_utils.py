"""
MXNet utility such as learning rate scheduler
"""


class MultiEpochScheduler(object):
    """
    Multi epoch scheduler
    """
    def __init__(self, epochs, factor=0.1):
        self.epochs = epochs
        self.factor = factor
        self.limit = epochs[0]

    def update_lr(self, trainer, cur_epoch):
        if cur_epoch >= self.limit:
            if len(self.epochs) > 0:
                self.epochs.pop(0)
                lr = trainer.learning_rate
                trainer.set_learning_rate(lr * self.factor)

                if len(self.epochs) > 0:
                    self.limit = self.epochs[0]

                return True
            else:
                return False
        else:
            return False
