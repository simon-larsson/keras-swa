""" Keras SWA Object
"""

import keras.backend as K
from keras.callbacks import Callback
from keras.layers import BatchNormalization
from .callback import create_swa_callback_class

SWA = create_swa_callback_class(K, Callback, BatchNormalization)
