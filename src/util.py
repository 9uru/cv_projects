'''
General utils
'''

import cv2

def start_capture(filename: str) -> cv2.VideoCapture:
    '''
    Return a camera capture if filename is none
    else a video file capture
    '''
    if filename is None:
        filename = 0

    return cv2.VideoCapture(0)
