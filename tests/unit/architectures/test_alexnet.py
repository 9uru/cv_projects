'''
Unit test for alexnet
'''

# pylint: disable=import-error
from src.architectures.alexnet import AlexNet



def test_alexnet():
    '''
    unit test for alenxet creation
    '''
    model = AlexNet((227, 227, 3), 10)
    assert model.count_params() > 6e6

    model = AlexNet((227, 227, 3), 2)
    assert model.layers[-1].activation.__name__ == 'sigmoid'
