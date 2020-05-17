'''
General utils
'''
from typing import Tuple
import numpy as np
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


def resize_dataset(
        image_data: np.ndarray,
        target_im_size: Tuple[int]) -> np.ndarray:
    '''
    Resize every image in dataset
    assumes 1st dimension is the image index
    '''
    data_resized = []
    for i in range(image_data.shape[0]):
        img = image_data[i, :, :]
        data_resized.append(
            cv2.resize(img, dsize=target_im_size, interpolation=cv2.INTER_CUBIC))
    data_resized = np.array(data_resized)
    return data_resized
