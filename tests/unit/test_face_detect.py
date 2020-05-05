'''
unit test for face detect
'''

from unittest.mock import MagicMock as Mock, patch
from src.scripts import face_detect


@patch('cv2.cvtColor')
@patch('cv2.resize')
def test_detect_draw_rect(cvtcol, resize):
    '''
    test for detect and draw
    '''
    face_cascade = Mock()
    frame = Mock()
    gray = Mock()
    cvtcol.return_value = gray
    resize.return_value = gray
    frame = face_detect.detect_draw_rect(face_cascade, frame)
    face_cascade.detectMultiScale.assert_called_with(gray, 1.1, 4)
