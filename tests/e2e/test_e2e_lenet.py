'''
E2E testing of LeNet with MNIST and Fashion MNIST
'''

# pylint: disable=import-error
import os
import datetime
import numpy as np
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.datasets import mnist
from tensorflow.keras import utils
from tensorflow.keras.callbacks import TensorBoard
from src.architectures.lenet import LeNet


def test_lenet_mnist():
    '''
    train and test lenet on mnist
    '''
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    _, height, width = x_train.shape
    print(type(x_train))

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
