'''
unit test for face detect
'''

from unittest.mock import MagicMock as Mock, patch
import numpy as np
from src.scripts import face_detect


@patch('cv2.cvtColor')
@patch('cv2.resize')
def test_detect_draw_rect_cascade(resize, cvtcol):
    '''
    test for detect and draw
    '''
    face_cascade = Mock()
    frame = Mock()
    gray = Mock()
    cvtcol.return_value = gray
    resize.return_value = gray
    frame = face_detect.detect_draw_rect_cascade(face_cascade, frame)
    face_cascade.detectMultiScale.assert_called_with(gray, 1.1, 4)



@patch('cv2.resize')
@patch('cv2.dnn.blobFromImage')
def test_detect_draw_rect_dnn(bbi, resize):
    '''
    test for detect and draw type dnn
    '''
    bbi = Mock()
    resize = Mock()
    frame = Mock()
    frame.shape = (100, 100, 3)
    blob = Mock()
    net = Mock()
    net.forward.return_value = np.zeros((1, 1))
    confidence = 0.5
    resize.return_value = blob
    bbi.return_value = blob
    frame = face_detect.detect_draw_rect_dnn(net, frame, confidence)
    net.forward.assert_called()
