'''
Face detector using nn
based on: https://www.pyimagesearch.com/2018/02/26/face-detection-with-opencv-and-deep-learning/
on webcam feed
'''
import os
import logging
import argparse
import cv2
import numpy as np
from src import util

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)


def detect_draw_rect_cascade(
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


def detect_draw_rect_dnn(
        net: cv2.dnn,
        frame: np.ndarray,
        confidence: float) -> np.ndarray:
    '''
    Detect faces and draw rects on provided image
    '''

    # convert frame to a blob
    (height, width) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)), 1.0,
        (300, 300), (104.0, 177.0, 123.0))

    # pass blob through network
    net.setInput(blob)
    detections = net.forward()

    if len(detections.shape) > 2:
        # loop over the detections
        for i in range(0, detections.shape[2]):
            # 3rd dimension is confidence use it to filter

            if detections[0, 0, i, 2] < confidence:
                continue

            # compute the bounding box of detected face
            box = detections[0, 0, i, 3:7] * np.array([
                width, height, width, height])
            (start_x, start_y, end_x, end_y) = box.astype('int')

            # draw the bounding box of the face along with the associated
            # probability
            text = '{:.2f}%'.format(confidence * 100)
            text_y = start_y - 10 if start_y - 10 > 10 else start_y + 10
            cv2.rectangle(
                frame,
                (start_x, start_y),
                (end_x, end_y),
                (0, 0, 255), 2)
            cv2.putText(
                frame,
                text,
                (start_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

    return frame


def main():  # pragma: no cover
    ''' Main workflow '''

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--detector',
        type=str,
        default='dnn',
        help='dnn or cascade')
    parser.add_argument(
        '-p', '--prototxt', required=True,
        help='path to Caffe deploy prototxt file')
    parser.add_argument(
        '-m', '--model', required=True,
        help='path to Caffe pre-trained model')
    parser.add_argument(
        '-c', '--confidence', type=float, default=0.5,
        help='minimum probability to filter weak detections')
    args = vars(parser.parse_args())

    logging.info('Loading model')
    if args['detector'] == 'dnn':
        net = cv2.dnn.readNetFromCaffe(args['prototxt'], args['model'])
    else:
        face_cascade_file = os.path.join(
            util.get_cv2_data_loc(),
            'haarcascade_frontalface_default.xml')
        face_cascade = cv2.CascadeClassifier(face_cascade_file)

    capture = util.start_capture(None)



    num_frames = 0
    while num_frames < 500:
        ret, frame = capture.read()
        if not ret:
            raise NameError('Could not retrieve a frame')

        if args['detector'] == 'dnn':
            frame = detect_draw_rect_dnn(net, frame, args['confidence'])
        else:
            frame = detect_draw_rect_cascade(face_cascade, frame)

        # Display the output
        cv2.imshow('img', frame)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        num_frames += 1


if __name__ == '__main__':  # pragma: no cover
    main()
