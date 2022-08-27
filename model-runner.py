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

model = keras.models.load_model("model.keras")

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def get_pose_info(image):
    results = pose.process(image)
    p_landmarks = results.pose_landmarks
    if not p_landmarks:
        return None
    x_pos = []
    y_pos = []

    for i in str(p_landmarks).split('landmark')[1:]:
        i_1 = i.split()
        x_pos.append(int(640 * float(i_1[2])))
        y_pos.append(int(480 * float(i_1[4])))

    return np.array(list(zip(x_pos, y_pos))), p_landmarks

# angle of a vector
def angle(v1: np.array, v2: np.array) -> float:
    return acos(np.dot(v1, v2) / norm(v1) / norm(v2))

def dist(v1: np.array, v2: np.array, v3: np.array = None) -> float:
    """
        If v3 is not defined, then this is equal to the distance from the line Origin-V1 to the point V2\n
        If v3 is defined, then this is equal to the distance from the line v1-v2 to v3
    """
    if v3 is not None:
        return dist(v2-v1, v3-v1)
    else:
        # cos = v1*v2/|v1|/|v2|; sin = sqrt(1-cos^2); dist=sin |v2|
        return sqrt(1 - (np.dot(v1, v2) / norm(v1) / norm(v2)) ** 2) * norm(v2)

def get_stats(vecs: np.array):
    # angle of the left elbow
    lef_elbow_angle = angle(vecs[11] - vecs[13], vecs[15] - vecs[13])
    # angle of the right elbow
    rig_elbow_angle = angle(vecs[12] - vecs[14], vecs[16] - vecs[14])
    torso_angle = (angle(vecs[25] - vecs[23], vecs[11] - vecs[23]) + angle(vecs[26] - vecs[24], vecs[12] - vecs[24])) / 2

    # torso dimensions
    torso = (vecs[24]-vecs[12], vecs[23]-vecs[11], vecs[12]-vecs[11], vecs[24]-vecs[23]) # right, left, top, bottom

    # torso height / torso width; useful for detecting if posture is sideways
    torso_ratio = (linalg.norm(torso[0]) + linalg.norm(torso[1])) / (linalg.norm(torso[2]) + linalg.norm(torso[3]))
    # average sine of the torso - measures the "angle" of the body
    torso_sin = (abs(torso[0][1] / linalg.norm(torso[0])) + abs(torso[1][1] / linalg.norm(torso[1]))) / 2

    # distance of the chest from the line formed by the fingers and toes
    chest_dis = linalg.norm(np.cross(vecs[20] - vecs[32], vecs[32] - vecs[12])) / linalg.norm(vecs[20] - vecs[32])

    return {
        "lef_ea": lef_elbow_angle,
        "rig_ea": rig_elbow_angle,
        "ta": torso_angle,
        "tr": torso_ratio,
        "ts": torso_sin,
        "cdis": chest_dis
    }

def preprocess(stats: dict) -> np.array:
    names = ['lef_ea', 'rig_ea', 'ta', 'tr', 'ts', 'cdis']
    return np.array([stats[i] for i in names])

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
        print(get_stats(vecs))

        image.flags.writeable = True

        input_vector = preprocess(get_stats(vecs)).reshape((1, -1))
        prediction = model(input_vector).numpy()[0, 0]

        image = cv2.putText(image, str("Pushup" if prediction > 0.5 else "No Pushup"), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

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
