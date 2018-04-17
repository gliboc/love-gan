from keras.models import Sequential
from keras.layers import (
    Dense,
    Activation,
    BatchNormalization,
    Conv2DTranspose,
    ZeroPadding2D,
    LeakyReLU
)

CONFIG = {
    'size': 64,   # size of generated pictures
    'ngf': 64,    # number of G filters for the first conv layer
    'nz': 100,    # dimension for Z
    'nc': 3,      # number of output channels
    'ndf': 64,    # number of D filters for the first conv layer
    'lr': 0.0002, # initial learning rate for adam
    'beta1': 0.5, # momentum term for adam
    'alpha': 0.2, # LeakyReLU slope parameter
}

# TODO: clean this up
def create_generator_model():
    model = Sequential()

    model.add(Conv2DTranspose(
        filters=CONFIG['ngf'] * 8,
        kernel_size=4,
        input_shape=(CONFIG['nz'],)
    ))

    model.add(BatchNormalization(
        activation='relu'
    ))

    for i in reversed(range(3)):

        model.add(Conv2DTranspose(
            filters=CONFIG['ngf'] * (2 ** i),
            kernel_size=4,
            strides=2,
            padding='same'
        ))

        model.add(BatchNormalization(
            activation='relu'
        ))

    model.add(Conv2DTranspose(
        filters=CONFIG['nc'],
        kernel_size=4,
        strides=2,
        padding='same',
        activation='tanh'
    ))

    return model


def create_descriminator_model():
    model = Sequential()

    model.add(Conv2D(
        filters=CONFIG['ndf'],
        kernel_size=4,
        padding='same',
        input_shape=(CONFIG['size'], CONFIG['size'], CONFIG['nc'])
    ))

    model.add(LeakyReLU(CONFIG['alpha']))

    model.add(Conv2D(
        filters=CONFIG['ndf'] * 2,
        kernel_size=4,
        padding='same'
    ))

    model.add(BatchNormalization())
    model.add(LeakyReLU(CONFIG['alpha']))

    model.add(Conv2D(
        filters=CONFIG['ndf'] * 4,
        kernel_size=4,
        padding='same'
    ))

    model.add(BatchNormalization())
    model.add(LeakyReLU(CONFIG['alpha']))

    model.add(Conv2D(
        filters=CONFIG['ndf'] * 8,
        kernel_size=4,
        padding='same'
    ))

    model.add(BatchNormalization())
    model.add(LeakyReLU(CONFIG['alpha']))

    model.add(Conv2D(
        filters=1,
        kernel_size=4,
        activation='sigmoid'
    ))

    return model
