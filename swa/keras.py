""" Keras SWA: callback utility for performing stochastic weight averaging (SWA).
"""

import keras.backend as K
from keras.callbacks import Callback
from keras.layers import BatchNormalization

class SWA(Callback):
    """ Stochastic Weight Averging.

    # Paper
        title: Averaging Weights Leads to Wider Optima and Better Generalization
        link: https://arxiv.org/abs/1803.05407

    # Arguments
        start_epoch:   integer, epoch when swa should start.
        lr_schedule:   string, type of learning rate schedule.
        swa_lr:        float, learning rate for swa.
        swa_lr2:       float, upper bound of cyclic learning rate.
        swa_freq:      integer, length of learning rate cycle.
        batch_size     integer, batch size (for batch norm with generator)
        verbose:       integer, verbosity mode, 0 or 1.
    """
    def __init__(self,
                 start_epoch,
                 lr_schedule='manual',
                 swa_lr=0.001,
                 swa_lr2=None,
                 swa_freq=1,
                 batch_size=None,
                 verbose=0):

        super(SWA, self).__init__()
        self.start_epoch = start_epoch - 1
        self.lr_schedule = lr_schedule
        self.swa_lr = swa_lr

        #if no user determined upper bound, make one based off of the lower bound
        self.swa_lr2 = (swa_lr2 if swa_lr2 != None else 10*swa_lr)
        self.swa_freq = swa_freq
        self.batch_size = batch_size
        self.verbose = verbose

        if start_epoch < 2:
            raise ValueError('"swa_start" attribute cannot be lower than 2.')

        schedules = ['manual', 'constant', 'cyclic']

        if self.lr_schedule not in schedules:
            raise ValueError('"{}" is not a valid learning rate schedule'
                             .format(self.lr_schedule))

        if self.lr_schedule == 'cyclic' and self.swa_freq < 2:
            raise ValueError(
                '"swa_freq" must be higher than 1 for cyclic schedule.')

        if self.lr_schedule == 'cyclic' and self.swa_lr > self.swa_lr2:
            raise ValueError('"swa_lr" must be lower than "swa_lr2".')

    def on_train_begin(self, logs=None):
        self.lr_record = []
        self.epochs = self.params.get('epochs')

        if self.start_epoch >= self.epochs - 1:
            raise ValueError(
                '"swa_start" attribute must be lower than "epochs".')

        self.init_lr = K.get_value(self.model.optimizer.lr)

        if self.init_lr < self.swa_lr:
            raise ValueError(
                '"swa_lr" must be lower than rate set in optimizer.')

        self._check_batch_norm()

        if self.has_batch_norm and self.batch_size is None:
            raise ValueError('"batch_size" needs to be set for the Keras API for '
                             'models with batch normalization.')

    def on_epoch_begin(self, epoch, logs=None):

        self._scheduler(epoch)

        if self.lr_schedule != 'cyclic':
            self._update_lr(epoch)
        self.current_epoch = epoch

        if self.is_swa_start_epoch:
            self.swa_weights = self.model.get_weights()

            if self.verbose > 0:
                print('\nEpoch %05d: starting stochastic weight averaging'
                      % (epoch + 1))

        if self.is_batch_norm_epoch:
            self._set_swa_weights(epoch)

            if self.verbose > 0:
                print('\nResetting batch normalization layers. This may take a moment.')
            self._reset_batch_norm()

            if self.verbose > 0:
                print('\nEpoch %05d: running forward pass to adjust batch normalization'
                      % (epoch + 1))

    def on_batch_begin(self, batch, logs=None):

        
        if self.lr_schedule == 'cyclic':
            self._update_lr(self.current_epoch, batch)

        if self.is_batch_norm_epoch:

            batch_size = self.batch_size
            momentum = batch_size / (batch*batch_size + batch_size)

            for layer in self.batch_norm_layers:
                layer.momentum = momentum

    def on_epoch_end(self, epoch, logs=None):

        if self.is_swa_start_epoch:
            self.swa_start_epoch = epoch

        if self.is_swa_epoch and not self.is_batch_norm_epoch:
            if self.verbose >0: 
                print('\nWeights being added to SWA.\n')
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

    def _update_lr(self, epoch, batch=None):

        if self.is_batch_norm_epoch:
            K.set_value(self.model.optimizer.lr, 0)
        elif self.lr_schedule == 'constant':
            lr = self._constant_schedule(epoch)
            K.set_value(self.model.optimizer.lr, lr)
        elif self.lr_schedule == 'cyclic':
            lr = self._cyclic_schedule(epoch, batch)
            self.lr_record.append(lr)
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

    def _cyclic_schedule(self, epoch, batch):
        """Designed after Section 3.1 of Averaging Weights Leads to
        Wider Optima and Better Generalization(https://arxiv.org/abs/1803.05407)
        """
        #mini-batches per epoch, equal to training_samples / batch_size
        steps = self.params.get('steps') 

        swa_epoch = (epoch - self.start_epoch) % self.swa_freq
        cycle_length = self.swa_freq * steps

        i = (swa_epoch * steps) + (batch + 1) #batch 0 indexed, so need to add 1
        if epoch >= self.start_epoch:

            t = (((i-1) % cycle_length) + 1)/cycle_length
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

            #to get properly initialized moving mean and moving variance weights
            #we initialize a new batch norm layer from the config of the existing
            #layer, build that layer, retrieve its moving mean and moving var weights
            #and then delete the layer
            new_batch_norm = BatchNormalization(**layer.get_config())
            new_batch_norm.build(layer.input_shape)
            new_gamma, new_beta, new_moving_mean, new_moving_var = new_batch_norm.get_weights()

            #now we can get rid of the new_batch_norm layer
            del new_batch_norm

            #get the trained gamma and beta from the layer
            trained_gamma, trained_beta, trained_moving_mean, trained_moving_var = layer.get_weights()

            #set weights
            layer.set_weights([trained_gamma, trained_beta,
                               new_moving_mean, new_moving_var])

            """layer.moving_mean = layer.add_weight(
                shape=shape,
                name='moving_mean',
                initializer=layer.moving_mean_initializer,
                trainable=layer.moving_mean.trainable)
            
            layer.moving_variance = layer.add_weight(
                shape=shape,
                name='moving_variance',
                initializer=layer.moving_variance_initializer,
                trainable=layer.moving_variance.trainable)"""

    def _restore_batch_norm(self):

        for layer, momentum in zip(self.batch_norm_layers, self.batch_norm_momentums):
            layer.momentum = momentum
