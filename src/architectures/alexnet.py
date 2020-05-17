'''
Alexnet implementation
'''

# Guru 5/2020

# pylint: disable=import-error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D


class AlexNet(Sequential):  # pylint: disable=too-few-public-methods
    '''
    AlexNet class
    '''
    def __init__(
            self,
            input_shape: tuple,
            num_classes: int):  # pylint: disable=R0801
        ''' Initialize model '''
        super().__init__()
        self.add(
            Conv2D(
                filters=96,
                kernel_size=(11, 11),
                strides=(4, 4),
                activation='relu',
                input_shape=input_shape,
                padding='valid')
        )
        self.add(
            MaxPooling2D(
                pool_size=(3, 3),
                strides=(2, 2),
                padding='valid'
            )
        )
        self.add(
            Conv2D(
                filters=256,
                kernel_size=(5, 5),
                strides=(1, 1),
                activation='relu',
                padding='same')
        )
        self.add(
            MaxPooling2D(
                pool_size=(3, 3),
                strides=(2, 2),
                padding='valid'
            )
        )
        self.add(
            Conv2D(
                filters=384,
                kernel_size=(3, 3),
                strides=(1, 1),
                activation='relu',
                padding='same')
        )
        self.add(
            Conv2D(
                filters=384,
                kernel_size=(3, 3),
                strides=(1, 1),
                activation='relu',
                padding='same')
        )
        self.add(
            Conv2D(
                filters=256,
                kernel_size=(3, 3),
                strides=(1, 1),
                activation='relu',
                padding='same')
        )
        self.add(
            MaxPooling2D(
                pool_size=(3, 3),
                strides=(2, 2),
                padding='valid'
            )
        )
        self.add(
            Flatten()
        )
        self.add(
            Dense(units=4096, activation='relu')
        )
        self.add(
            Dense(units=4096, activation='relu')
        )
        if num_classes == 2:
            self.add(Dense(units=1, activation='sigmoid'))
        else:
            self.add(Dense(units=num_classes, activation='softmax'))
