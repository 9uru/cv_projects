'''
E2E testing of LeNet with MNIST and Fashion MNIST
'''

# pylint: disable=import-error
import os
import datetime
from typing import Tuple
import numpy as np
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.datasets import mnist
from tensorflow.keras import utils
from tensorflow.keras.callbacks import TensorBoard
from src.architectures.lenet import LeNet
from src.architectures.alexnet import AlexNet
from src import util


def load_preprocess_mnist(
        target_im_size: Tuple[int]) -> Tuple[np.ndarray]:
    '''
    Load and preprocess mnist dataset
    '''
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = util.resize_dataset(x_train, target_im_size)
    x_test = util.resize_dataset(x_test, target_im_size)
    _, height, width = x_train.shape

    # Set numeric type to float32 from uint8
    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)

    # Normalize value to [0, 1]
    x_train /= 255
    x_test /= 255

    # Transform lables to one-hot encoding
    y_train = utils.to_categorical(y_train, 10)
    y_test = utils.to_categorical(y_test, 10)

    # Reshape the dataset into 4D array
    x_train = x_train.reshape(x_train.shape[0], height, width, 1)
    x_test = x_test.reshape(x_test.shape[0], height, width, 1)

    return x_train, y_train, x_test, y_test


def test_lenet_mnist():
    '''
    train and test lenet on mnist
    '''

    x_train, y_train, x_test, y_test = load_preprocess_mnist(
        target_im_size=(28, 28))

    model = LeNet(x_train[0].shape, 10)

    model.compile(
        loss=categorical_crossentropy,
        optimizer='SGD',
        metrics=['accuracy']
    )

    log_dir = os.path.join(
        "logs\\fit\\",
        datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    # Specify the callback object
    tensorboard_callback = TensorBoard(
        log_dir=log_dir,
        histogram_freq=1)

    model.fit(
        x_train,
        y=y_train,
        epochs=20,
        validation_data=(x_test, y_test),
        callbacks=[tensorboard_callback],
        verbose=0)


def test_alexnet_mnist():
    '''
    train and test lenet on mnist
    '''

    x_train, y_train, x_test, y_test = load_preprocess_mnist(
        target_im_size=(227, 227))

    model = AlexNet(x_train[0].shape, 10)

    model.compile(
        loss=categorical_crossentropy,
        optimizer='SGD',
        metrics=['accuracy']
    )

    log_dir = os.path.join(
        "logs\\fit\\",
        datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    # Specify the callback object
    tensorboard_callback = TensorBoard(
        log_dir=log_dir,
        histogram_freq=1)

    model.fit(
        x_train,
        y=y_train,
        epochs=20,
        validation_data=(x_test, y_test),
        callbacks=[tensorboard_callback],
        verbose=0)
