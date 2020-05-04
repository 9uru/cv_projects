'''
Utils unit tests
'''

import cv2
from src import util


def test_start_capture():
    '''
    test capture methods
    '''
    cap = util.start_capture(None)
    assert isinstance(cap, cv2.VideoCapture)