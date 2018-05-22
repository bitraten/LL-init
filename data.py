import tensorflow as tf


class Dataset:
    def __init__(self, name, to_float=True, to_categorical=True):
        self.name = name
        self.to_float = to_float
        self.to_categorical = to_categorical

    def load_data(self, x_train, y_train, x_test, y_test):
        if self.to_float:
            self.x_train = x_train.astype('float32') / 255
            self.x_test = x_test.astype('float32') / 255
        else:
            self.x_train = x_train
            self.x_test = x_test

        if self.to_categorical:
            self.y_train = tf.keras.utils.to_categorical(y_train)
            self.y_test = tf.keras.utils.to_categorical(y_test)
        else:
            self.y_train = y_train
            self.y_test = y_test

        self.input_shape = x_train.shape[1:4]


def load_mnist():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    INPUT_SHAPE = x_train.shape[1:4] + (1,)
    x_train = x_train.reshape((60000,) + INPUT_SHAPE)
    x_test = x_test.reshape((10000,) + INPUT_SHAPE)

    mnist = Dataset("mnist", to_float=True, to_categorical=True)
    mnist.load_data(x_train, y_train, x_test, y_test)

    return mnist

def load_cifar10():

    # TODO CIFAR10 vs CIFAR100
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    cifar10 = Dataset("cifar10", to_float=True, to_categorical=True)
    cifar10.load_data(x_train, y_train, x_test, y_test)

    return cifar10

def load_imagenet():
    pass

