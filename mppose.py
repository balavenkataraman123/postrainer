import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
ismoving = 0
lastnotmovedy = 0
lastnotmovedx = 0
distanceconficence = 20

cap = cv2.VideoCapture(0)
with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      continue


    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    p_landmarks = results.pose_landmarks
    if p_landmarks:
      xpos = []
      ypos = []
      vis = []
      for i in str(p_landmarks).split('landmark')[1:]:
        i_1 = i.split()
        xpos.append(int(640 *  float(i_1[2])))
        ypos.append(int(480 *  float(i_1[4])))
        vis.append(float(i_1[8]))
      print(min(vis))
      minx = min(xpos)
      maxx = max(xpos)
      miny = min(ypos)
      maxy = max(ypos)
      image.flags.writeable = True
      mp_drawing.draw_landmarks(
          image,
          results.pose_landmarks,
          mp_pose.POSE_CONNECTIONS,
          landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
      
    cv2.imshow("Image", image)

    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
