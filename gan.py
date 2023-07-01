from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
from six.moves import urllib

import tensorflow.compat.v2.feature_column as fc
import tensorflow_datasets as tfds
import tensorflow as tf

def build_generator():
    model = Sequential()

    # Takes in random values and reshapes it to 7x7x128
    # Beginnings of a generated image
    #we use 7 7 because it will simplify the reshape later
    model.add(Dense(7*7*128, input_dim=128))
    model.add(LeakyReLU(0.2))
    model.add(Reshape((7,7,128)))#doesnt have neurone just reshaping the data 
    # Upsampling block 1 
    model.add(UpSampling2D()) #after this layer 7x7 will be 
    model.add(Conv2D(128, 5, padding='same'))
    model.add(LeakyReLU(0.2))


    return model