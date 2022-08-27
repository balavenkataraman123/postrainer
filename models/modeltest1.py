from cmath import e
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
from preprocessing import *
from PIL import Image

motivquotes = [
" \"The last three or four reps is what makes the muscle grow. This area of pain divides a champion from someone who is not a champion \"-Arnold Schwarzenegger", 
" \"The successful warrior is the average man, with laser-like focus.\" -Bruce Lee",
" \"The pain you feel today will be the strength you feel tomorrow.\" -Arnold Schwarzenegger",
" \"You miss 100% of the shots you don\'t take.\" -Wayne Gretzky",
"\"Blood, sweat and respect. First two you give. Last one you earn.\" -The Rock"
]

model = keras.models.load_model("pushup-model.keras")

# define the camera
cap = cv2.VideoCapture(0)

numpushup = 0
currstate = 1 # 1 for up and 0 for down
predictionaverage = 0
past40cdis = []

exercise = 0
justchanged = 1

exercises = ["PUSHUPS"]
reps = [10]
jctimer = 0
thismotquote = ""
minvis = 1
while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue
    if exercises[exercise] == "PUSHUPS":
        if justchanged == 1:
            jctimer = 200
            thismotquote = random.choice(motivquotes)
            justchanged = 0
        if jctimer == 0:
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            parts = get_pose_info(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image = cv2.resize(image, (1280, 960), interpolation=cv2.INTER_AREA)
            if parts is not None:
                vecs, landmarks = parts
                stats = (Pushup.get_stats(vecs))
                cdis = stats["cdis"]
                trsa = stats["ta"]
                image.flags.writeable = True    
                input_vector = Pushup.preprocess(vecs).reshape((1, -1))
                
                prediction = model(input_vector).numpy()[0, 0]
                image = cv2.putText(image, str(numpushup), (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

                if minvis <= 0.2:
                    image = cv2.putText(image, "The camera cannot see you well", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                if prediction > 0.5:
                    image = cv2.copyMakeBorder(image ,20,20,20,20,cv2.BORDER_CONSTANT,value=[0, 255, 0])
                    print(cdis)
                    past40cdis.append(cdis)
                    if len(past40cdis) > 40:
                        past40cdis.pop(0)
                    if cdis <= 110 and currstate == 1:
                        currstate = 0
                    if cdis >= 190 and currstate == 0:
                        currstate = 1
                        numpushup += 1
                    if numpushup == reps[exercise]:
                        justchanged = 1
                        exercise += 1
                    if min(past40cdis) > 110:
                        image = cv2.putText(image, "YOU NEED TO GO LOWER!!! ", (20, 200), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 0, 255), 2, cv2.LINE_AA)
                    if len(past40cdis) > 20 and trsa < 2:
                        image = cv2.putText(image, "KEEP YOUR BACK STRAIGHT!!!! ", (20, 240), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 0, 255), 2, cv2.LINE_AA)

                    image = cv2.putText(image, "YOUR SCORE: " + str(prediction) , (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                else:
                    image = cv2.copyMakeBorder(image ,20,20,20,20,cv2.BORDER_CONSTANT,value=[0, 0, 255])

            mp_drawing.draw_landmarks(
                image,
                landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        else:
            jctimer -= 1
            image = cv2.resize(image, (1280, 960), interpolation=cv2.INTER_AREA)
            image = cv2.putText(image, "DO " + str(reps[exercise]) + " " + exercises[exercise], (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            if exercises[exercise] == "PUSHUPS":
                image = cv2.putText(image, "put your laptop on the floor, about 6 feet away from you", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA,)    
                image = cv2.putText(image, "do your pushup parallel to your screen", (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA,)    

            image = cv2.putText(image, thismotquote, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA,)
    cv2.imshow("Image", image)
    if cv2.waitKey(5) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()
