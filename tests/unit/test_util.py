'''
Utils unit tests
'''
# pylint: disable=no-member
import numpy as np
import cv2
from src import util


def test_start_capture():
    '''
    test capture methods
    '''
    cap = util.start_capture(None)
    assert isinstance(cap, cv2.VideoCapture)


def test_get_cv2_data_loc():
    '''
    test cv2 data loc acquisition
    '''
    assert isinstance(util.get_cv2_data_loc(), str)


def test_resize_dataset():
    '''
    test resizing every image in a dataset
    '''
    data_before = np.random.rand(10, 200, 200)
    data_after = util.resize_dataset(
        data_before,
        (300, 300))
    assert data_after.shape == (10, 300, 300)
