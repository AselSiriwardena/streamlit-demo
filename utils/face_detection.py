import face_recognition
import numpy as np
import cv2
import streamlit as st


def detect(image):
    cv2.imwrite('org.jpg', image)

    rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    rgb_img = cv2.resize(rgb_img, None, fx=0.5, fy=0.5)
    bbs = face_recognition.face_locations(rgb_img)

    for bb in bbs:
        (t, r, b, l) = np.array(bb, dtype='int') * 2
        cv2.rectangle(image, (l, t), (r, b), (0, 255, 0), 2)

    return image
