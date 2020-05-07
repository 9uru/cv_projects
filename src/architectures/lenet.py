'''
Lenet implementation using TF
source: https://engmrk.com/lenet-5-a-classic-cnn-architecture/
'''

# Guru 5/2020

# pylint: disable=import-error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, AveragePooling2D


class LeNet(Sequential):  # pylint: disable=too-few-public-methods
    '''
    LeNet 5
    '''
    def __init__(
            self,
            input_shape: tuple,
            num_classes: int):
        ''' Initialize model '''
        super().__init__()
        self.add(
            Conv2D(
                filters=6,
                kernel_size=(5, 5),
                strides=(1, 1),
                activation='tanh',
                input_shape=input_shape,
                padding='valid')
        )

        self.add(
            AveragePooling2D(
                pool_size=(2, 2),
                strides=(2, 2),
                padding='valid'
            )
        )

        self.add(
            Conv2D(
                filters=16,
                kernel_size=(5, 5),
                strides=(1, 1),
                activation='tanh',
                padding='valid'
            )
        )

        self.add(
            AveragePooling2D(
                pool_size=(2, 2),
                strides=(2, 2),
                padding='valid'
            )
        )

        self.add(Flatten())
        self.add(Dense(units=120, activation='tanh'))
        self.add(Dense(units=84, activation='tanh'))
        if num_classes == 2:
            self.add(Dense(units=1, activation='sigmoid'))
        else:
            self.add(Dense(units=num_classes, activation='softmax'))
