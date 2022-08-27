import cv2
import pickle
import mediapipe as mp
import numpy as np
import numpy.linalg as linalg
from numpy.linalg import norm
import matplotlib.pyplot as plt
from math import cos, sin, acos, asin, sqrt
from preprocessing import *

def collect_frames(name: str, ratio: float):
    cap = cv2.VideoCapture(name)
    frames = []
    while cap.isOpened():
        ret, image = cap.read()
        if not ret:
            break

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        frames.append(image)

    return frames[:int(len(frames) * ratio)]

file = "data/no-pushup-0.mp4"

frames = collect_frames(file, 1)
features = [Pushup.get_stats(get_pose_info(i)[0]) for i in frames if get_pose_info(i) and not get_pose_info(i) is None]

with open(f"{file}.pkl", "wb") as f:
    pickle.dump(features, f)