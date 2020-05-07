'''
Unit test for lenet
'''

# pylint: disable=import-error
from src.architectures.lenet import LeNet


def test_lenet():
    '''
    unit test for lenet creation
    '''
    model = LeNet((32, 32, 1), 10)
    assert model.count_params() == 61706

    model = LeNet((32, 32, 1), 2)
    assert model.count_params() == 60941
