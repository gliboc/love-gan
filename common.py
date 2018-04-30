# -*- coding: utf-8 -*-


from glob import glob
from math import ceil

from PIL import Image
import numpy as np
from numpy.random import randint, normal
from keras.models import Model, Input
from keras.optimizers import Adam


def load_image(path):
    return (np.array(Image.open(path)).astype(np.float32) - 127.5) / 127.5


def load_images(pat):
    return np.array([load_image(p) for p in glob(pat)])


def save_image(arr, path):
    img = Image.fromarray((127.5 * arr + 127.5).astype(np.uint8), mode='RGB')
    img.save(path)


def save_mosaic(cols, arr, path):
    n, w, h, _ = arr.shape
    rows = ceil(n / cols)

    imgs = (127.5 * arr + 127.5).astype(np.uint8)
    mosaic = Image.new('RGB', (2 + cols * (w + 2), 2 + rows * (h + 2)))
    for i in range(n):
        mosaic.paste(Image.fromarray(imgs[i,:,:,:], mode='RGB'),
                     (2 + (i % cols) * (w + 2), 2 + (i // cols) * (h + 2)))

    mosaic.save(path)


class GAN:
    def __init__(self, gen, discr, lr=0.0002, beta_1=0.5):
        opt = Adam(lr=lr, beta_1=beta_1)
        self.gen = gen
        self.discr = discr
        self.nz = gen.input_shape[1]
        
        self.discr.compile(loss='binary_crossentropy', optimizer=opt,
                           metrics=['accuracy'])
        
        self.discr.trainable = False
        z = Input(shape=(self.nz,))
        self.comb = Model(z, self.discr(self.gen(z)))
        self.comb.compile(loss='binary_crossentropy', optimizer=opt)

    def train_on_batch(self, real):
        half_batch = real.shape[0]
        d_loss_r = self.discr.train_on_batch(real, np.ones((half_batch, 1)))

        fake = self.gen.predict(normal(0, 1, (half_batch, self.nz)))
        d_loss_f = self.discr.train_on_batch(fake, np.zeros((half_batch, 1)))

        z = normal(0, 1, (2 * half_batch, self.nz))
        g_loss = self.comb.train_on_batch(z, np.ones((2 * half_batch, 1)))

        return .5 * np.add(d_loss_r, d_loss_f), g_loss

    def train(self, epochs, batch_size, save_interval, src_pat, out_pat):
        x_train = load_images(src_pat)
        print('#####', x_train.shape)
        try:
            for e in range(epochs):
                real = x_train[randint(0, x_train.shape[0], batch_size)]
                d_loss, g_loss = self.train_on_batch(real)
                print('[E %04i] [D loss: %.3f, acc: %.2f%%] [G loss: %.3f]' %
                      (e, d_loss[0], d_loss[1] * 100, g_loss))
                if e % save_interval == 0:
                    z = normal(0, 1, (25, self.nz))
                    save_mosaic(5, self.gen.predict(z), out_pat % e)
        except KeyboardInterrupt:
            pass
