from __future__ import print_function, division

from keras.utils.vis_utils import plot_model
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.utils import to_categorical
import keras.backend as K

from glob import glob
from PIL import Image

import numpy as np

# Config table for adjusting the hyper-parameters
CONFIG = {
    'size' : 96,
    'nc' : 3,
    'nclasses' : 20,
    'g_input_dim' : 82,
    'lr' : 0.002
}


class INFOGAN():
    def __init__(self, **cfg):
        self.img_rows = cfg['size']
        self.img_cols = cfg['size']
        self.channels = cfg['nc']
        self.num_classes = cfg['nclasses']
        self.img_shape = (cfg['size'], cfg['size'], cfg['nc'])
        self.latent_dim = cfg['g_input_dim']

        # According to the paper, Adam works best as an optimizer
        optimizer = Adam(cfg['lr'], 0.5)

        # The discriminator and recognition network share most of their architecture
        # Therefore, they are built a the same time
        self.discriminator, self.recognition = self.build_discr_and_rec()

        # The discriminator optimizes its valid guesses
        self.discriminator.compile(loss=['binary_crossentropy'],
                                    optimizer=optimizer,
                                    metrics=['accuracy'])

        # The recognition network maximizes the mutual information loss
        self.recognition.compile(loss=[self.mutual_information_loss],
                                    optimizer=optimizer,
                                    metrics=['accuracy'])

        self.generator = self.build_generator()

        # The input of the generator is concatenated from noise and Q's output
        gen_input = Input(shape=(self.latent_dim,))
        img = self.generator(gen_input)

        self.discriminator.trainable = False

        valid = self.discriminator(img)
        target_label = self.recognition(img)

        # The combined model. The generator and the recognition network are both
        # trained at the same time to fool the discriminator AND maximimize mutual information
        self.comb = Model(gen_input, [valid, target_label])
        self.comb.compile(loss=['binary_crossentropy', self.mutual_information_loss],
            optimizer=optimizer)


    def build_generator(self):

        model = Sequential()

        # We use dimension (6, 6, 256) because we want to obtain 96x96-shaped images,
        # and 96 = 6 * (2 ** k), where k is the depth of the network in terms of
        # convolutionnal layers: 4.
        # We use 256 neurons as in the Chairs implementation, in the InfoGAN paper
        model.add(Dense(256 * 6 * 6, activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((6, 6, 256)))
        model.add(BatchNormalization(momentum=0.8))

        model.add(UpSampling2D())
        model.add(Conv2D(256, kernel_size=3, padding="same"))
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

        model.add(UpSampling2D())
        model.add(Conv2D(32, kernel_size=3, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Conv2D(self.channels, kernel_size=3, padding='same'))
        model.add(Activation("tanh"))

        gen_input = Input(shape=(self.latent_dim,))
        img = model(gen_input)

        print("Generator:")
        model.summary()
        plot_model(model, to_file='model_plots/generator_plot.png', show_shapes=True, show_layer_names=True)

        return Model(gen_input, img)


    def build_discr_and_rec(self):

        img = Input(shape=self.img_shape)

        # Shared layers between discriminator and recognition network
        model = Sequential()

        model.add(Conv2D(64, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(256, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(512, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Flatten())

        img_embedding = model(img)

        # Discriminator
        validity = Dense(1, activation='sigmoid')(img_embedding)

        # Recognition network
        q_net = Dense(128, activation='relu')(img_embedding)
        label = Dense(self.num_classes, activation='softmax')(q_net)

        # Visualize models
        print("Discriminator:")
        m_val = Model(img, validity)
        m_val.summary()

        print("Recognition:")
        m_lab = Model(img, label)
        m_lab.summary()

        plot_model(model, to_file='model_plots/discriminator_plot.png', show_shapes=True, show_layer_names=True)
        plot_model(model, to_file='model_plots/recognition_plot.png', show_shapes=True, show_layer_names=True)
        # Return discriminator and recognition network
        return (m_val, m_lab)


    def mutual_information_loss(self, c, c_given_x):
        """The mutual information metric we aim to minimize"""
        eps = 1e-8
        conditional_entropy = K.mean(- K.sum(K.log(c_given_x + eps) * c, axis=1))
        entropy = K.mean(- K.sum(K.log(c + eps) * c, axis=1))

        return conditional_entropy + entropy



    def sample_generator_input(self, batch_size):
        # Random noise
        noise = np.random.normal(0, 1, (batch_size, 62))

        # Randomized classes between 0 and num_classes
        labels = np.random.randint(0, self.num_classes, batch_size).reshape(-1, 1)
        labels = to_categorical(labels, num_classes=self.num_classes)

        return noise, labels


    def train(self, epochs, batch_size=128, sample_interval=50):

        # Load the dataset
        X_train = self.load_images()

        # Rescale -1 to 1
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        print(X_train)

        half_batch = int(batch_size / 2)

        for epoch in range(epochs):

            # Training discriminator
            idx = np.random.randint(0, X_train.shape[0], half_batch)
            imgs = X_train[idx]

            noise, labels = self.sample_generator_input(half_batch)
            gen_input = np.concatenate((noise, labels), axis=1)
            gen_imgs = self.generator.predict(gen_input)

            valid = np.ones((half_batch, 1))
            fake = np.zeros((half_batch, 1))

            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)

            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # Now training generator and classifier
            valid = np.ones((batch_size, 1))
            noise, labels = self.sample_generator_input(batch_size)
            gen_input = np.concatenate((noise, labels), axis=1)

            g_loss = self.comb.train_on_batch(gen_input, [valid, labels])

            # Progress
            print ("%d [D loss: %.2f, acc.: %.2f%%] [Q loss: %.2f] [G loss: %.2f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss[1], g_loss[2]))

            if epoch % sample_interval == 0:
                self.sample_images(epoch)


    def load_images(self):
        xs = []

        for f in glob('./input/*.jpg'):
            img = np.array(Image.open(f))
            if img.shape == self.img_shape:
                xs.append(img)
            else:
                print('bad shape for %s: %s' % (f, img.shape))

        return np.array(xs)






    def save_imgs(self, epoch, n=(10,20)):
        """ Print an output to visualize progress """
        s0 = self.img_rows + 2
        s1 = self.img_cols + 2

        out = Image.new('RGB', (2 + n[0] * s0, 2 + n[1] * s1))


        for i in range(n[0]):
            noise, _ = self.sample_generator_input(n[0])
            label = to_categorical(np.full(fill_value=i, shape=(n[1],1)), num_classes=self.num_classes)
            gen_input = np.concatenate((noise, label), axis=1)
            imgs = self.generator.predict(gen_input)
            imgs = np.array(128 * imgs + 128, dtype=np.uint8)
            for j in range(n[1]):
                img = imgs[n[1]*i + j,:,:,:]
                out.paste(Image.fromarray(img, mode='RGB'), (2 + s0 * i, 2 + s1 * j))

        out.save('images/fruits_%04i.png' % epoch)


    def save_model(self):

        def save(model, model_name):
            model_path = "saved_model/%s.json" % model_name
            weights_path = "saved_model/%s_weights.hdf5" % model_name
            options = {"file_arch": model_path,
                        "file_weight": weights_path}
            json_string = model.to_json()
            open(options['file_arch'], 'w').write(json_string)
            model.save_weights(options['file_weight'])

        save(self.generator, "generator")
        save(self.discriminator, "discriminator")
        save(self.recognition, "classifier")
        save(self.comb, "adversarial")


if __name__ == "__main__":
    d = INFOGAN(**CONFIG)
    d.train(500, 128, 2)

