'''
Utils unit tests
'''
# pylint: disable=no-member
import cv2
from src import util


def test_start_capture():
    '''
    test capture methods
    '''
    cap = util.start_capture(None)
    assert isinstance(cap, cv2.VideoCapture)
