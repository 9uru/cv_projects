'''
Face detector using haar cascade
on webcam feed
'''

import os
import cv2
import numpy as np
from src import util


def detect_draw_rect(
        face_cascade: cv2.CascadeClassifier,
        frame: np.ndarray) -> np.ndarray:
    '''
    Detect faces and draw rects on provided image
    '''

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (300, 300))

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Draw rectangle around the faces
    for (left, top, width, height) in faces:
        cv2.rectangle(
            frame,
            (left, top),
            (left + width, top + height),
            (255, 0, 0), 2)

    return frame


def main():  # pragma: no cover
    ''' Main workflow '''
    capture = util.start_capture(None)
    face_cascade_file = os.path.join(
        util.get_cv2_data_loc(),
        'haarcascade_frontalface_default.xml')
    face_cascade = cv2.CascadeClassifier(face_cascade_file)

    num_frames = 0
    while num_frames < 500:
        ret, frame = capture.read()
        if not ret:
            raise NameError('Could not retrieve a frame')

        frame = detect_draw_rect(face_cascade, frame)

        # Display the output
        cv2.imshow('img', frame)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        num_frames += 1


if __name__ == '__main__':  # pragma: no cover
    main()
