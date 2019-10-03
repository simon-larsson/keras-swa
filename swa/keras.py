""" Keras SWA: callback utility for performing stochastic weight averaging (SWA).
"""

import keras.backend as K
from keras.callbacks import Callback

class SWA(Callback):
    """ Stochastic Weight Averging.

    # Paper
        title: Averaging Weights Leads to Wider Optima and Better Generalization
        link: https://arxiv.org/abs/1803.05407

    # Arguments
        start_epoch: integer, epoch when swa should start.
        lr_schedule: string, kind of learning rate schedule.
                        'optimizer': optimizer handles learning rate.
                        'constant': learning rate will go from 'lr' to 'swa_lr' with 
                                    a constant decay.
        swa_lr: float, minimum learning rate.
        verbose: integer, verbosity mode, 0 or 1.
    """
    def __init__(self, start_epoch, lr_schedule='optimizer', swa_lr=0.001, verbose=0):
        super(SWA, self).__init__()
        self.start_epoch = start_epoch - 1
        self.lr_schedule = lr_schedule
        self.swa_lr = swa_lr
        self.verbose = verbose
        
        if start_epoch < 2:
            raise ValueError('"swa_start" attribute cannot be lower than 2.')

    def on_train_begin(self, logs=None):
        self.epochs = self.params.get('epochs')
        
        if self.start_epoch >= self.epochs - 1:
            raise ValueError('"swa_start" attribute must be lower than "epochs".')

        self.init_lr = K.get_value(self.model.optimizer.lr)

        if self.init_lr < self.swa_lr:
            raise ValueError('"swa_lr" must be lower than rate set in optimizer.')

    def on_epoch_begin(self, epoch, logs=None):
        self._update_lr(epoch)

        if epoch == self.start_epoch:
            self.swa_weights = self.model.get_weights()

            if self.verbose > 0:
                print('Epoch %05d: starting stochastic weight averaging'
                      % (epoch + 1))

    def on_epoch_end(self, epoch, logs=None):
        if epoch >= self.start_epoch:
            self.swa_weights = [(swa_w * (epoch - self.start_epoch) + w)
                                / ((epoch - self.start_epoch) + 1)
                                        for swa_w, w in zip(self.swa_weights,
                                                            self.model.get_weights())]

    def on_train_end(self, logs=None):
        self.model.set_weights(self.swa_weights)
        if self.verbose > 0:
            print('\nEpoch %05d: final model weights set to stochastic weight average'
                  % (self.epochs))

    def _update_lr(self, epoch):
        if self.lr_schedule == 'constant':
            lr = self._constant_schedule(epoch)
            K.set_value(self.model.optimizer.lr, lr)

    def _constant_schedule(self, epoch):
        t = epoch / self.start_epoch
        lr_ratio = self.swa_lr / self.init_lr
        if t <= 0.5:
            factor = 1.0
        elif t <= 0.9:
            factor = 1.0 - (1.0 - lr_ratio) * (t - 0.5) / 0.4
        else:
            factor = lr_ratio
        return self.init_lr * factor
