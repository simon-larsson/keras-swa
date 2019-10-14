""" TF-Keras SWA: callback utility for performing stochastic weight averaging (SWA).
"""

import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import BatchNormalization

class SWA(Callback):
    """ Stochastic Weight Averging.

    # Paper
        title: Averaging Weights Leads to Wider Optima and Better Generalization
        link: https://arxiv.org/abs/1803.05407

    # Arguments
        start_epoch:   integer, epoch when swa should start.
        lr_schedule:   string, type of learning rate schedule.
        swa_lr:        float, learning rate for swa.
        batch_size:    integer, training batch size, not always required.
        verbose:       integer, verbosity mode, 0 or 1.
    """
    def __init__(self, start_epoch, lr_schedule=None, 
                 swa_lr=0.001, batch_size=None, verbose=0):
        
        super(SWA, self).__init__()
        self.start_epoch = start_epoch - 1
        self.lr_schedule = lr_schedule
        self.swa_lr = swa_lr
        self.batch_size = batch_size
        self.verbose = verbose
        
        if start_epoch < 2:
            raise ValueError('"swa_start" attribute cannot be lower than 2.')
            
        schedules = ['none', 'constant']
        
        if self.lr_schedule is None:
            self.lr_schedule = 'none'
        
        if self.lr_schedule not in schedules:
            raise ValueError('"{}" is not a valid learning rate schedule' \
                             .format(self.lr_schedule))

    def on_train_begin(self, logs=None):
        
        self.epochs = self.params.get('epochs')
        
        if self.params.get('batch_size') is not None:
            self.batch_size = self.params.get('batch_size')
        
        if self.start_epoch >= self.epochs - 1:
            raise ValueError('"swa_start" attribute must be lower than "epochs".')

        self.init_lr = K.get_value(self.model.optimizer.lr)

        if self.init_lr < self.swa_lr:
            raise ValueError('"swa_lr" must be lower than rate set in optimizer.')
            
        self._check_batch_norm()
        
        if self.has_batch_norm and self.batch_size is None:
            raise ValueError('"batch_size" has to be set manually for models with ' 
                             'batch normalization when fitting with generators.')

    def on_epoch_begin(self, epoch, logs=None):
        
        self._update_lr(epoch)

        if epoch == self.start_epoch:
            self.swa_weights = self.model.get_weights()

            if self.verbose > 0:
                print('Epoch %05d: starting stochastic weight averaging'
                      % (epoch + 1))
                
        if epoch == self.epochs - 1 and self.has_batch_norm:
            self._set_swa_weights(epoch)
            self._reset_batch_norm()
            self.running_bn_epoch = True
            
            if self.verbose > 0:
                print('\nEpoch %05d: running forward pass to adjust batch normalization'
                      % (self.epochs))

    def on_batch_begin(self, batch, logs=None):
        
        if self.running_bn_epoch:
            
            momentum = self.batch_size / (batch*self.batch_size + self.batch_size)

            for layer in self.batch_norm_layers:
                layer.momentum = momentum
                
    def on_epoch_end(self, epoch, logs=None):
        
        if epoch >= self.start_epoch and not self.running_bn_epoch:
            self.swa_weights = [(swa_w * (epoch - self.start_epoch) + w)
                                / ((epoch - self.start_epoch) + 1)
                                        for swa_w, w in zip(self.swa_weights,
                                                            self.model.get_weights())]

    def on_train_end(self, logs=None):
        
        if not self.has_batch_norm:
            self._set_swa_weights(self.epochs - 1)
        else:
            self._restore_batch_norm()

    def _update_lr(self, epoch):  
        
        if epoch == self.epochs - 1 and self.has_batch_norm:   
            K.set_value(self.model.optimizer.lr, 0)
        elif self.lr_schedule == 'constant':
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
    
    def _set_swa_weights(self, epoch):
        
        self.model.set_weights(self.swa_weights)
        
        if self.verbose > 0:
            print('\nEpoch %05d: final model weights set to stochastic weight average'
                  % (epoch + 1))     
    
    def _check_batch_norm(self):
              
        self.batch_norm_momentums = []
        self.batch_norm_layers = []
        self.has_batch_norm = False
        self.running_bn_epoch = False
        
        for layer in self.model.layers:
            if issubclass(layer.__class__, BatchNormalization):
                self.has_batch_norm = True
                self.batch_norm_momentums.append(layer.momentum)
                self.batch_norm_layers.append(layer)
                    
        if self.verbose > 0 and self.has_batch_norm:
            print('Model uses batch normalization. SWA will require last epoch '
                  'to be a forward pass and will run with no learning rate')
    
    def _reset_batch_norm(self):
        
        for layer in self.batch_norm_layers:
            shape = (list(layer.input_spec.axes.values())[0],)

            layer.moving_mean = layer.add_weight(
                shape=shape,
                name='moving_mean',
                initializer=layer.moving_mean_initializer,
                trainable=layer.moving_mean.trainable)
            
            layer.moving_variance = layer.add_weight(
                shape=shape,
                name='moving_variance',
                initializer=layer.moving_variance_initializer,
                trainable=layer.moving_variance.trainable)
          
    def _restore_batch_norm(self):
        
        for layer, momentum in zip(self.batch_norm_layers, self.batch_norm_momentums):
            layer.momentum = momentum
