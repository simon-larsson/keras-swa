""" SWA Callback
"""


def create_swa_callback_class(K, callback, batch_normalization):
    """Injecting library dependencies"""

    class SWA(callback):
        """Stochastic Weight Averging.

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

        def __init__(
            self,
            start_epoch,
            lr_schedule="manual",
            swa_lr="auto",
            swa_lr2="auto",
            swa_freq=1,
            batch_size=None,
            verbose=0,
        ):

            super().__init__()
            self.start_epoch = start_epoch - 1
            self.lr_schedule = lr_schedule
            self.swa_lr = swa_lr

            # if no user determined upper bound, make one based off of the lower bound
            self.swa_lr2 = swa_lr2 if swa_lr2 is not None else 10 * swa_lr
            self.swa_freq = swa_freq
            self.batch_size = batch_size
            self.verbose = verbose

            if start_epoch < 2:
                raise ValueError('"swa_start" attribute cannot be lower than 2.')

            schedules = ["manual", "constant", "cyclic"]

            if self.lr_schedule not in schedules:
                raise ValueError(
                    '"{}" is not a valid learning rate schedule'.format(
                        self.lr_schedule
                    )
                )

            if self.lr_schedule == "cyclic" and self.swa_freq < 2:
                raise ValueError(
                    '"swa_freq" must be higher than 1 for cyclic schedule.'
                )

            if self.swa_lr == "auto" and self.swa_lr2 != "auto":
                raise ValueError(
                    '"swa_lr2" cannot be manually set if "swa_lr" is automatic.'
                )

            if (
                self.lr_schedule == "cyclic"
                and self.swa_lr != "auto"
                and self.swa_lr2 != "auto"
                and self.swa_lr > self.swa_lr2
            ):
                raise ValueError('"swa_lr" must be lower than "swa_lr2".')

        def on_train_begin(self, logs=None):

            self.lr_record = []
            self.epochs = self.params.get("epochs")

            if self.start_epoch >= self.epochs - 1:
                raise ValueError('"swa_start" attribute must be lower than "epochs".')

            self.init_lr = K.eval(self.model.optimizer.lr)

            # automatic swa_lr
            if self.swa_lr == "auto":
                self.swa_lr = 0.1 * self.init_lr

            if self.init_lr < self.swa_lr:
                raise ValueError('"swa_lr" must be lower than rate set in optimizer.')

            # automatic swa_lr2 between initial lr and swa_lr
            if self.lr_schedule == "cyclic" and self.swa_lr2 == "auto":
                self.swa_lr2 = self.swa_lr + (self.init_lr - self.swa_lr) * 0.25

            self._check_batch_norm()

            if self.has_batch_norm and self.batch_size is None:
                raise ValueError(
                    '"batch_size" needs to be set for models with batch normalization layers.'
                )

        def on_epoch_begin(self, epoch, logs=None):

            self.current_epoch = epoch
            self._scheduler(epoch)

            # constant schedule is updated epoch-wise
            if self.lr_schedule == "constant":
                self._update_lr(epoch)

            if self.is_swa_start_epoch:
                self.swa_weights = self.model.get_weights()

                if self.verbose > 0:
                    print(
                        "\nEpoch %05d: starting stochastic weight averaging"
                        % (epoch + 1)
                    )

            if self.is_batch_norm_epoch:
                self._set_swa_weights(epoch)

                if self.verbose > 0:
                    print(
                        "\nEpoch %05d: reinitializing batch normalization layers"
                        % (epoch + 1)
                    )

                self._reset_batch_norm()

                if self.verbose > 0:
                    print(
                        "\nEpoch %05d: running forward pass to adjust batch normalization"
                        % (epoch + 1)
                    )

        def on_batch_begin(self, batch, logs=None):

            # update lr each batch for cyclic lr schedule
            if self.lr_schedule == "cyclic":
                self._update_lr(self.current_epoch, batch)

            if self.is_batch_norm_epoch:

                batch_size = self.batch_size
                momentum = batch_size / (batch * batch_size + batch_size)

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

            for batch_lr in self.lr_record:
                self.model.history.history.setdefault("lr", []).append(batch_lr)

        def _scheduler(self, epoch):

            swa_epoch = epoch - self.start_epoch

            self.is_swa_epoch = (
                epoch >= self.start_epoch and swa_epoch % self.swa_freq == 0
            )
            self.is_swa_start_epoch = epoch == self.start_epoch
            self.is_batch_norm_epoch = epoch == self.epochs - 1 and self.has_batch_norm

        def _average_weights(self, epoch):

            return [
                (swa_w * ((epoch - self.start_epoch) / self.swa_freq) + w)
                / ((epoch - self.start_epoch) / self.swa_freq + 1)
                for swa_w, w in zip(self.swa_weights, self.model.get_weights())
            ]

        def _update_lr(self, epoch, batch=None):

            if self.is_batch_norm_epoch:
                lr = 0
                K.set_value(self.model.optimizer.lr, lr)
            elif self.lr_schedule == "constant":
                lr = self._constant_schedule(epoch)
                K.set_value(self.model.optimizer.lr, lr)
            elif self.lr_schedule == "cyclic":
                lr = self._cyclic_schedule(epoch, batch)
                K.set_value(self.model.optimizer.lr, lr)
            self.lr_record.append(lr)

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
            # steps are mini-batches per epoch, equal to training_samples / batch_size
            steps = self.params.get("steps")

            # occasionally steps parameter will not be set. We then calculate it ourselves
            if steps is None:
                steps = self.params["samples"] // self.params["batch_size"]

            swa_epoch = (epoch - self.start_epoch) % self.swa_freq
            cycle_length = self.swa_freq * steps

            # batch 0 indexed, so need to add 1
            i = (swa_epoch * steps) + (batch + 1)
            if epoch >= self.start_epoch:

                t = (((i - 1) % cycle_length) + 1) / cycle_length
                return (1 - t) * self.swa_lr2 + t * self.swa_lr
            else:
                return self._constant_schedule(epoch)

        def _set_swa_weights(self, epoch):

            self.model.set_weights(self.swa_weights)

            if self.verbose > 0:
                print(
                    "\nEpoch %05d: final model weights set to stochastic weight average"
                    % (epoch + 1)
                )

        def _check_batch_norm(self):

            self.batch_norm_momentums = []
            self.batch_norm_layers = []
            self.has_batch_norm = False
            self.running_bn_epoch = False

            for layer in self.model.layers:
                if issubclass(layer.__class__, batch_normalization):
                    self.has_batch_norm = True
                    self.batch_norm_momentums.append(layer.momentum)
                    self.batch_norm_layers.append(layer)

            if self.verbose > 0 and self.has_batch_norm:
                print(
                    "Model uses batch normalization. SWA will require last epoch "
                    "to be a forward pass and will run with no learning rate"
                )

        def _reset_batch_norm(self):

            for layer in self.batch_norm_layers:

                # to get properly initialized moving mean and moving variance weights
                # we initialize a new batch norm layer from the config of the existing
                # layer, build that layer, retrieve its reinitialized moving mean and
                # moving var weights and then delete the layer
                bn_config = layer.get_config()
                new_batch_norm = batch_normalization(**bn_config)
                new_batch_norm.build(layer.input_shape)
                new_moving_mean, new_moving_var = new_batch_norm.get_weights()[-2:]
                # get rid of the new_batch_norm layer
                del new_batch_norm
                # get the trained gamma and beta from the current batch norm layer
                trained_weights = layer.get_weights()
                new_weights = []
                # get gamma if exists
                if bn_config["scale"]:
                    new_weights.append(trained_weights.pop(0))
                # get beta if exists
                if bn_config["center"]:
                    new_weights.append(trained_weights.pop(0))
                new_weights += [new_moving_mean, new_moving_var]
                # set weights to trained gamma and beta, reinitialized mean and variance
                layer.set_weights(new_weights)

        def _restore_batch_norm(self):

            for layer, momentum in zip(
                self.batch_norm_layers, self.batch_norm_momentums
            ):
                layer.momentum = momentum

    return SWA
