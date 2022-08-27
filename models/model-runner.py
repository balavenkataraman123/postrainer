import cv2
import mediapipe as mp
import numpy as np
import numpy.linalg as linalg
from numpy.linalg import norm
import matplotlib.pyplot as plt
from math import cos, sin, acos, asin, sqrt
import tensorflow as tf
from tensorflow import keras
from keras import layers
import random
from PIL import Image
from preprocessing import *

pushup_model = keras.models.load_model("pushup-model.keras")
situp_model = keras.models.load_model("situp-model.keras")
squat_model = keras.models.load_model("squat-model.keras")
curl_model = keras.models.load_model("curl-model.keras")

# define the camera
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    parts = get_pose_info(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if parts is not None:
        vecs, landmarks = parts

        image.flags.writeable = True

        pushup_prediction = pushup_model(Pushup.preprocess(vecs).reshape((1, -1))).numpy()[0, 0]
        situp_prediction = situp_model(Situp.preprocess(vecs).reshape((1, -1))).numpy()[0, 0]
        squat_prediction = squat_model(Squat.preprocess(vecs).reshape((1, -1))).numpy()[0, 0]
        curl_prediction = curl_model(BicepCurl.preprocess(vecs).reshape((1, -1))).numpy()[0, 0]

        image = cv2.putText(image, str("Pushup" if pushup_prediction > 0.5 else "No Pushup"), (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        image = cv2.putText(image, str("Situp" if situp_prediction > 0.5 else "No Situp"), (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        image = cv2.putText(image, str("Squat" if squat_prediction > 0.5 else "No Squat"), (20, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        image = cv2.putText(image, str("Bicep Curl" if squat_prediction > 0.5 else "No Bicep Curl"), (20, 160),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        mp_drawing.draw_landmarks(
            image,
            landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

    cv2.imshow("Image", image)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
