'''
General utils
'''

import cv2


def get_cv2_data_loc() -> str:
    '''
    return full path to where cv2 data is stored
    '''
    return '\\'.join(cv2.__file__.split('\\')[:-1]) + '\\data'


def start_capture(filename: str) -> cv2.VideoCapture:
    '''
    Return a camera capture if filename is none
    else a video file capture
    '''
    if filename is None:
        filename = 0

    return cv2.VideoCapture(0)
