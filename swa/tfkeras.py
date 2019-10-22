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
        swa_lr:        float, learning rate for swa sampling.
        swa_lr2:       float, upper bound of cyclic learning rate.
        swa_freq:      integer, length of learning rate cycle.
        verbose:       integer, verbosity mode, 0 or 1.
    """
    def __init__(self, 
                 start_epoch, 
                 lr_schedule='manual', 
                 swa_lr=0.001, 
                 swa_lr2=0.003,
                 swa_freq=1,
                 verbose=0):
        
        super(SWA, self).__init__()
        self.start_epoch = start_epoch - 1
        self.lr_schedule = lr_schedule
        self.swa_lr = swa_lr
        self.swa_lr2 = swa_lr2
        self.swa_freq = swa_freq
        self.verbose = verbose
        
        if start_epoch < 2:
            raise ValueError('"swa_start" attribute cannot be lower than 2.')
            
        schedules = ['manual', 'constant', 'cyclic']
        
        if self.lr_schedule not in schedules:
            raise ValueError('"{}" is not a valid learning rate schedule' \
                             .format(self.lr_schedule))

        if self.lr_schedule == 'cyclic' and self.swa_freq < 2:
            raise ValueError('"swa_freq" must be higher than 1.')

        if self.lr_schedule == 'cyclic' and self.swa_lr > self.swa_lr2:
            raise ValueError('"swa_lr" must be lower than "swa_lr2".')

    def on_train_begin(self, logs=None):
        
        self.epochs = self.params.get('epochs')
        
        if self.start_epoch >= self.epochs - 1:
            raise ValueError('"swa_start" attribute must be lower than "epochs".')

        self.init_lr = K.get_value(self.model.optimizer.lr)

        if self.init_lr < self.swa_lr:
            raise ValueError('"swa_lr" must be lower than rate set in optimizer.')
            
        self._check_batch_norm()

    def on_epoch_begin(self, epoch, logs=None):
        
        self._scheduler(epoch)        
        self._update_lr(epoch)       

        if self.is_swa_start_epoch:
            self.swa_weights = self.model.get_weights()
            
            if self.verbose > 0:
                print('\nEpoch %05d: starting stochastic weight averaging'
                      % (epoch + 1))
                
        if self.is_batch_norm_epoch:
            self._set_swa_weights(epoch)
            self._reset_batch_norm()
            
            if self.verbose > 0:
                print('\nEpoch %05d: running forward pass to adjust batch normalization'
                      % (epoch + 1))

    def on_batch_begin(self, batch, logs=None):
        
        if self.is_batch_norm_epoch:
            
            batch_size = self.params['samples']      
            momentum = batch_size / (batch*batch_size + batch_size)

            for layer in self.batch_norm_layers:
                layer.momentum = momentum
                
    def on_epoch_end(self, epoch, logs=None):

        if self.is_swa_start_epoch:
            self.swa_start_epoch = epoch
        
        if self.is_swa_epoch and not self.is_batch_norm_epoch:
            self.swa_weights = self._average_weights(epoch)

    def on_train_end(self, logs=None):
        
        if not self.has_batch_norm:
            self._set_swa_weights(self.epochs)
        else:
            self._restore_batch_norm()
    
    def _scheduler(self, epoch):
        
        swa_epoch = (epoch - self.start_epoch)
        
        self.is_swa_epoch = epoch >= self.start_epoch and swa_epoch % self.swa_freq == 0
        self.is_swa_start_epoch = epoch == self.start_epoch
        self.is_batch_norm_epoch = epoch == self.epochs - 1 and self.has_batch_norm
    
    def _average_weights(self, epoch):

        return [(swa_w * (epoch - self.start_epoch) + w)
                / ((epoch - self.start_epoch) + 1)
                    for swa_w, w in zip(self.swa_weights, self.model.get_weights())]

    def _update_lr(self, epoch):  
        
        if self.is_batch_norm_epoch:
            K.set_value(self.model.optimizer.lr, 0)
        elif self.lr_schedule == 'constant':
            lr = self._constant_schedule(epoch)
            K.set_value(self.model.optimizer.lr, lr)
        elif self.lr_schedule == 'cyclic':
            lr = self._cyclic_schedule(epoch)
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
    
    def _cyclic_schedule(self, epoch):
        
        swa_epoch = epoch - self.start_epoch
        
        if epoch >= self.start_epoch:
            t = (((swa_epoch-1) % self.swa_freq)+1)/self.swa_freq
            return (1-t)*self.swa_lr2 + t*self.swa_lr
        else:
            return self._constant_schedule(epoch)
    
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
