# Keras SWA - Stochastic Weight Averaging

[![PyPI version](https://badge.fury.io/py/keras-swa.svg)](https://pypi.python.org/pypi/keras-swa/) 
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/simon-larsson/keras-swa/blob/master/LICENSE)

This is an implemention of SWA for Keras and TF-Keras. It is limited to only performing weight averaging and does not implement a learning rate scheduler.

## Introduction
Stochastic weight averaging (SWA) build upon the same principle as [snapshot ensembling](https://arxiv.org/abs/1704.00109) and [fast geometric ensembling](https://arxiv.org/abs/1802.10026). The idea is that averaging select stages of training can lead to better models. Where as the two former methods average by ensembling sampled models, SWA instead average weights. This has been shown to give comparable improvements but with just one model as a result.

[![Illustration](https://raw.githubusercontent.com/simon-larsson/keras-swa/master/swa_illustration.png)](https://raw.githubusercontent.com/simon-larsson/keras-swa/master/swa_illustration.png)

## Paper
 - Title: Averaging Weights Leads to Wider Optima and Better Generalization
 - Link: https://arxiv.org/abs/1803.05407
 - Authors: Pavel Izmailov, Dmitrii Podoprikhin, Timur Garipov, Dmitry Vetrov, Andrew Gordon Wilson
 - Repo: https://github.com/timgaripov/swa (PyTorch)

## Installation

    pip install keras-swa
    
### SWA

Keras callback object for SWA.  

#### Arguments
`swa_epochs` - The number of epochs in the end of training where SWA is applied.

`verbose` - Verbosity mode, 0 or 1.
    
#### Example

For Keras
```python
from sklearn.datasets.samples_generator import make_blobs
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense

from swa.keras import SWA
 
# make dataset
X, y = make_blobs(n_samples=1000, 
                  centers=3, 
                  n_features=2, 
                  cluster_std=2, 
                  random_state=2)

y = to_categorical(y)

# build model
model = Sequential()
model.add(Dense(50, input_dim=2, activation='relu'))
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

epochs = 10
swa_epochs = 4

# define swa callback
swa = SWA(swa_epochs=swa_epochs, verbose=1)

# train
model.fit(X, y, epochs=epochs, verbose=1, callbacks=[swa])
```

Or for Keras in Tensorflow

```python
from sklearn.datasets.samples_generator import make_blobs
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from swa.tfkeras import SWA
 
# make dataset
X, y = make_blobs(n_samples=1000, 
                  centers=3, 
                  n_features=2, 
                  cluster_std=2, 
                  random_state=2)

y = to_categorical(y)

# build model
model = Sequential()
model.add(Dense(50, input_dim=2, activation='relu'))
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy', 
              optimizer='adam', 
              metrics=['acc'])

epochs = 10
swa_epochs = 4

# define swa callback
swa = SWA(swa_epochs=swa_epochs, verbose=1)

# train
model.fit(X, y, epochs=epochs, verbose=1, callbacks=[swa])
```

Output:
```
Epoch 1/10
1000/1000 [==============================] - 0s 100us/step - loss: 1.1633
Epoch 2/10
1000/1000 [==============================] - 0s 37us/step - loss: 0.8161
Epoch 3/10
1000/1000 [==============================] - 0s 32us/step - loss: 0.7271
Epoch 4/10
1000/1000 [==============================] - 0s 31us/step - loss: 0.6771
Epoch 5/10
1000/1000 [==============================] - 0s 32us/step - loss: 0.6438
Epoch 6/10
1000/1000 [==============================] - 0s 31us/step - loss: 0.6183
Epoch 7/10
1000/1000 [==============================] - 0s 31us/step - loss: 0.5986

Epoch 00007: starting stochastic weight averaging
Epoch 8/10
1000/1000 [==============================] - 0s 28us/step - loss: 0.5789
Epoch 9/10
1000/1000 [==============================] - 0s 30us/step - loss: 0.5615
Epoch 10/10
1000/1000 [==============================] - 0s 31us/step - loss: 0.5472

Epoch 00010: final model weights set to stochastic weight average
```
