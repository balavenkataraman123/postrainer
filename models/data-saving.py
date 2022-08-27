# Just adding a few features here - I'm hoping for a common sense baseline to be made here

import cv2
import mediapipe as mp
import time
from preprocessing import *

frames = []

# define the camera
cap = cv2.VideoCapture(0)

time.sleep(2)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    frames.append(image)
    parts = get_pose_info(image)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if parts is not None:
        vecs, landmarks = parts

        mp_drawing.draw_landmarks(image, landmarks, mp_pose.POSE_CONNECTIONS, landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

    cv2.imshow("Image", image)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

video = cv2.VideoWriter("squat-0.mp4", 0, 15, (640, 480))

for i in range(len(frames)):
    parts = get_pose_info(frames[i])
    if parts is not None:
        vecs, landmarks = parts
        video.write(cv2.cvtColor(frames[i], cv2.COLOR_RGB2BGR))


