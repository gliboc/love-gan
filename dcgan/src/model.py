"""
Contains a basic definition of a DCGAN
Using keras-adversarial package
"""
import matplotlib as mpl

# This line allows mpl to run with no DISPLAY defined
mpl.use('Agg')

from keras.layers import Dense, Reshape, Flatten, Dropout, LeakyReLU,
Input, Activation, BatchNormalization
from keras.models import Sequential, Model
from keras.layers.convolutional import Convolution2D, UpSampling2D
from keras.optimizers import Adam
from keras.regularizers import l1, l1l2
from keras.datasets import mnist

import pandas as pd
import numpy as np

# specific gan modules
from keras_adversarial import AdversarialModel, ImageGridCallback,
simple_gan, gan_targets
from keras_adversarial import AdversarialOptimizerSimultaneous,
normal_latent_sampling, AdversarialOptimizerAlternating
from image_utils import dim_ordering_fix, dim_ordering_input,
dim_ordering_reshape, dim_ordering_unfix

def generator_model():
    model = Sequential()
    model.add(Dense(input_dim=100, output_dim=1024))
    model.add(Activation('tanh'))
    model.add(Dense(128*7*7))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(Reshape((128, 7, 7), input_shape=(128*7*7,)))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Convolution2D(64, 5, 5, border_mode='same'))
    model.add(Activation('tanh'))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Convolution2D(1, 5, 5, border_mode='same'))
    model.add(Activation('tanh'))
    return model

def discriminator_model():
    model = Sequential()
    model.add(Convolution2D(64, 5, 5, border_mode='same', \
            input_shape=(1, 28, 28)))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(128, 5, 5))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('tanh'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model

def gan_targets(n):
    """
    Training targets [gen_fake, gen_real, discr_fake, discr_real] = [1, 0, 0, 1]
    :param n: number of samples
    :return: array of targets
    """
    gen_fake = np.ones((n, 1))
    gen_real = np.zeros((n, 1))
    discr_fake = np.zeros((n, 1))
    discr_real = np.ones((n, 1))

    return [gen_fake, gen_real, discr_fake, discr_real]

def model_generator():
    nch = 256
    g_input = Input(shape=[100])
    H = Dense(nch * 14 * 14, init='glorot_normal')(g_input)
    H = BatchNormalization(mode=2)(H)
    H = Activation('relu')(H)
    H = dim_ordering_reshape(nch, 14)(H)
    H = UpSampling2D(size=(2, 2))(H)
    H = Convolution2D(int(nch / 2), 3, 3, border_mode='same', \
        init='glorot_uniform')(H)
    H = BatchNormalization(mode=2, axis=1)(H)
    H = Activation('relu')(H)
    H = Convolution2D(int(nch / 4), 3, 3, border_mode='same', \
        init='glorot_uniform')(H)
    H = BatchNormalization(mode=2, axis=1)(H)
    H = Activation('relu')(H)
    H = Convolution2D(1, 1, 1, border_mode='same', \
        init='glorot_uniform')(H)
    g_V = Activation('sigmoid')(H)
    return Model(g_input, g_V)

if __name__ == '__main__':
    gan_g = generator_model()
    gan_d = discriminator_model()
    
    adversarial_model = AdversarialModel(player_models = \
            [gan_g, gan_d], player_params=[generator.trainable_weights, \
            discriminator.trainable_weights], \
            player_names = ["generator", "discriminator"]


