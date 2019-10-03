""" TF-Keras SWA: callback utility for performing stochastic weight averaging (SWA).
"""

from tensorflow.keras.callbacks import Callback

class SWA(Callback):
    """ Stochastic Weight Averging.
    
    # Paper
        title: Averaging Weights Leads to Wider Optima and Better Generalization
        link: https://arxiv.org/abs/1803.05407
    
    # Arguments
        swa_epochs: integer, the number epochs swa will be applied to.
                    first swa epoch will be epochs - swa_epochs.
        verbose: integer, verbosity mode, 0 or 1.
    """
    def __init__(self, swa_epochs, verbose=0):
        super(SWA, self).__init__()
        self.swa_epochs = swa_epochs
        self.verbose = verbose
            
    def on_train_begin(self, logs=None):
        self.epochs = self.params.get('epochs')
        
        if self.swa_epochs < 1:
            raise ValueError('"swa_epochs" attribute cannot be lower than 1.')
        
        if self.swa_epochs >= self.epochs:
            raise ValueError('"swa_epochs" attribute must be lower than "epochs".')

        self.start_epoch = self.epochs - self.swa_epochs
    
    def on_epoch_begin(self, epoch, logs=None):
        if epoch == self.start_epoch:
            self.swa_weights = self.model.get_weights()
            
            if self.verbose > 0:
                print('\nEpoch %05d: starting stochastic weight averaging'
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
