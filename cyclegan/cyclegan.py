#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from glob import glob


from PIL import Image
from keras.utils.vis_utils import plot_model
from numpy import zeros, ones, array
from numpy.random import randint

from keras.models import Sequential, Model
from keras.layers import (Activation, BatchNormalization, LeakyReLU, Dense, Reshape,
                          Conv2D, Conv2DTranspose, Flatten, Input, UpSampling2D)
from keras.optimizers import Adam

import numpy as np


CONFIG = {
    'size': 32,   # size of generated pictures
    'ngf': 16,    # number of G filters for the first conv layer
    'nx': 100,    # dimension for X
    'ny': 100,    # dimension for Y
    'nc': 3,      # number of output channels
    'ndf': 64,    # number of D filters for the first conv layer
    'lr': 0.0002, # initial learning rate for adam
    'beta1': 0.5, # momentum term for adam
    'alpha': 0.2, # LeakyReLU slope parameter
    'lp':  0.0001 # Lambda parameter
}

class CYCLEGAN:
    def __init__(self, **cfg):
        self.img_shape = (cfg['size'], cfg['size'], cfg['nc'])
        self.nx = cfg['nx']
        self.ny = cfg['ny']
        self.ngf, self.ndf = cfg['ngf'], cfg['ndf']
        self.alpha = cfg['alpha']
        self.channels = cfg['nc']
        self.nc = cfg['nc']
        self.lp = cfg['lp']

        optimizer = Adam(lr=cfg['lr'], beta_1=cfg['beta1'])

        self.D_Y = self.build_discriminator()
        self.D_Y.compile(loss='mean_squared_error', optimizer=optimizer,
                            metrics=['accuracy'])
        self.D_Y.trainable = False

        self.D_X = self.build_discriminator()
        self.D_X.compile(loss='mean_squared_error', optimizer=optimizer,
                            metrics=['accuracy'])
        self.D_X.trainable = False

        self.G = self.build_generator()
        self.F = self.build_generator()

        x = Input(shape=self.img_shape)
        y = Input(shape=self.img_shape)

        G_x = self.G(x)
        F_y = self.F(y)

        F_G_x = self.F(G_x)
        G_F_y = self.G(F_y)

        valid_Y = self.D_Y(G_x)
        valid_X = self.D_X(F_y)

        self.combined = Model (inputs=[x, y],
                outputs=[valid_Y, valid_X,
                         F_G_x, G_F_y,
                         G_x, F_y])

        self.combined.compile (loss=['mean_squared_error',
                                    'mean_squared_error',
                                    'mean_absolute_error',
                                    'mean_absolute_error',
                                    'mean_absolute_error',
                                    'mean_absolute_error'],
                                loss_weights=[1, 1,
                                            self.lp, self.lp,
                                            self.lp, self.lp],
                                optimizer=optimizer)

        self.combined.summary()

    def train(self, epochs, batch_size, save_interval):
        # TODO: load images
        x_train = self.load_images('dog')
        y_train = self.load_images('cat')

        half_batch = batch_size // 2

        labels_true = np.ones((batch_size,))
        labels_false = np.zeros((batch_size,))

        for epoch in range(epochs):
            ## Discriminator
            print("Epoch %i" % epoch)

            real_x = x_train[randint(0, x_train.shape[0], half_batch)]
            real_y = y_train[randint(0, y_train.shape[0], half_batch)]

            d_loss_r1 = self.D_Y.train_on_batch(real_y, ones((half_batch, 1)))
            d_loss_r2 = self.D_X.train_on_batch(real_x, ones((half_batch, 1)))

            fake_Y = self.G.predict(real_x)
            d_loss_f1 = self.D_Y.train_on_batch(fake_Y, zeros((half_batch, 1)))

            fake_X = self.F.predict(real_y)
            d_loss_f2 = self.D_X.train_on_batch(fake_X, zeros((half_batch, 1)))

            print("Discriminator losses:", d_loss_r1, d_loss_r2, d_loss_f1, d_loss_f2)

            #print('[%04i] [D loss: %.3f, acc: %.2f%%]' %
            #     (epoch, d_loss[0], 100*d_loss[1]))

            print("\nGlobal GAN")
            g_loss = self.combined.train_on_batch([real_x, real_y],
                                                  [labels_true, labels_true,
                                                   real_x, real_y,
                                                   real_x, real_y])

            if epoch % save_interval == 0:
                self.save_images('cats', real_x, epoch)
                self.save_images('dogs', real_y, epoch)

    def load_images(self, animals):
        xs = []

        for path in glob('input/' + animals + '.*.jpg'):
            img = array(Image.open(path))
            if img.shape == self.img_shape:
                xs.append(img)
            else:
                print('bad shape for %s: %s' % (path, img.shape))

        return array(xs)

    def save_images(self, name, inpt, epoch, n=(5, 5)):
        if name == 'cats':
            imgs = self.gen1.predict(inpt)
        else:
            imgs = self.gen2.predict(inpt)

        # size of a tile
        s0 = self.img_shape[0] + 2
        s1 = self.img_shape[1] + 2

        mode = 'RGB'
        try:
            out = Image.new('RGB', (2 + n[0]*s0, 2 + n[1]*s1))
        except:
            mode = 'rgb'
            out = Image.new('rgb', (2 + n[0]*s0, 2 + n[1]*s1))

        for i in range(n[0]):
            for j in range(n[1]):
                out.paste(Image.fromarray(imgs[n[1]*i + j,:,:,:], mode=mode),
                          (2 + s0*i, 2 + s1*j))

        out.save('output/' + name + '/cyclegan_%04i.png' % epoch)

    def build_generator(self):

        model = Sequential()

        # We use dimension (6, 6, 256) because we want to obtain 96x96-shaped images,
        # and 96 = 6 * (2 ** k), where k is the depth of the network in terms of
        # convolutionnal layers: 4.
        # We use 256 neurons as in the Chairs implementation, in the InfoGAN paper

        model.add(Conv2D(128, kernel_size=4, strides=2, input_shape=self.img_shape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Conv2D(128, kernel_size=4, strides=2, input_shape=self.img_shape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))

        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))

        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Conv2D(self.channels, kernel_size=3, padding='same'))
        model.add(Activation("tanh"))

        gen_input = Input(shape=self.img_shape)
        img = model(gen_input)

        print("Generator:")
        model.summary()
        plot_model(model, to_file='model_plots/generator_plot.png', show_shapes=True, show_layer_names=True)

        return Model(gen_input, img)

    def build_discriminator(self):
        """Create the discriminator model as described in the paper."""

        model = Sequential()

        model.add(Conv2D(filters=self.ndf, kernel_size=4, padding='same',
                         strides=2, input_shape=self.img_shape))
        model.add(LeakyReLU(self.alpha))

        model.add(Conv2D(filters=self.ndf, kernel_size=4, padding='same',
                         strides=2))
        model.add(LeakyReLU(self.alpha))
        model.add(BatchNormalization())

        model.add(Conv2D(filters=self.ndf * 8, kernel_size=4, padding='same',
                         strides=2))
        model.add(LeakyReLU(self.alpha))
        model.add(BatchNormalization())

        model.add(Conv2D(filters=1, kernel_size=4, activation='sigmoid'))
        model.add(Flatten())

        print('\n%%% Discriminator %%%')
        model.summary()

        return model


if __name__ == "__main__":
    cy = CYCLEGAN(**CONFIG)
    cy.train(10, 64, 2)
