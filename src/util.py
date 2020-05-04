'''
General utils
'''

import cv2


def start_capture(filename):
    '''
    Return a camera capture if filename is none
    else a video file capture
    '''
    if filename is not None:
        return cv2.VideoCapture(filename)
    else:
        return cv2.VideoCapture(0)
