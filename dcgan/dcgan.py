#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from keras.models import Sequential, Model
from keras.layers import (Dense, Activation, BatchNormalization, LeakyReLU,
                          Conv2D, Conv2DTranspose, Flatten, Reshape)

from common import GAN


def generator_paper(nz=100, ngf=64, channels=3):
    """Create the generator model as described in the paper."""

    model = Sequential()

    model.add(Reshape((1, 1, nz), input_shape=(nz,)))
    model.add(Conv2DTranspose(filters=ngf * 8, kernel_size=4))

    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Conv2DTranspose(filters=ngf * 4, kernel_size=4,
                              strides=2, padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Conv2DTranspose(filters=ngf * 2, kernel_size=4,
                              strides=2, padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Conv2DTranspose(filters=ngf, kernel_size=4,
                              strides=2, padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Conv2DTranspose(filters=channels, kernel_size=4,
                              strides=2, padding='same',
                              activation='tanh'))
    return model


def discriminator_paper(size=64, channels=3, ndf=64, alpha=0.2):
    """Create the discriminator model as described in the paper."""

    model = Sequential()

    model.add(Conv2D(filters=ndf, kernel_size=4, padding='same', strides=2,
                     input_shape=(size, size, channels)))
    model.add(LeakyReLU(alpha))
    model.add(BatchNormalization())

    model.add(Conv2D(filters=ndf * 2, kernel_size=4, padding='same',
                     strides=2))
    model.add(LeakyReLU(alpha))
    model.add(BatchNormalization())

    model.add(Conv2D(filters=ndf * 4, kernel_size=4, padding='same',
                     strides=2))
    model.add(LeakyReLU(alpha))
    model.add(BatchNormalization())

    model.add(Conv2D(filters=ndf * 8, kernel_size=4, padding='same',
                     strides=2))
    model.add(LeakyReLU(alpha))
    model.add(BatchNormalization())

    model.add(Conv2D(filters=1, kernel_size=4, activation='sigmoid'))
    model.add(Flatten())

    return model


if __name__ == '__main__':
    gen, discr = generator_paper(), discriminator_paper()
    dcgan = GAN(gen, discr)
    dcgan.train(epochs=200, batch_size=100, save_interval=10,
                src_pat='input/hands_64/*.jpg',
                out_pat='output/dcgan%04i.jpg')
