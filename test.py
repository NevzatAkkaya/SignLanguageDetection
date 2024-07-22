import os
import numpy as np
import sklearn.metrics
import tensorflow as tf
import cv2 as cv
import time
import mediapipe as mp
import sklearn
from matplotlib import pyplot as plt
mp_holistic=mp.solutions.holistic
mp_drawings=mp.solutions.drawing_utils
actions=np.array(["thanks", "hello", "i love you"])
def mp_Detection(image, model):
    image=cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image.flags.writeable=False
    results=model.process(image)
    image.flags.writeable=True
    image=cv.cvtColor(image, cv.COLOR_RGB2BGR)
    return image, results

def draw_landmark(image, results):
    mp_drawings.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)
    mp_drawings.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    mp_drawings.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawings.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

def extract(results):
    face=np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(1404)
    rhand=np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(63)
    lhand=np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(63)
    pose=np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)
    return np.concatenate([pose, face, lhand, rhand])
sequence=[]
sentence=[]
threshold=0.4
cap=cv.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        isCap, frame = cap.read()
        image, results=mp_Detection(frame, holistic)
        draw_landmark(image, results)
        keypoints=extract(results)
        sequence.insert(0, keypoints)
        sequence=sequence[:40]
        if len(sequence)==30:
            res=model.predict(np.expand_dims(sequence, axis=0))[0]
        if res[np.argmax(res)] > threshold:
            if len(sentence)>0:
                if actions[np.argmax(res)] != sentence[-1]:
                    sentence.append(actions[np.argmax(res)])
                else:
                    sentence.append(actions[np.argmax(res)])
        if len(sentence)>5:
            sentence=sentence[-5:]        
        cv.rectangle(image, (0,0), (640,40), (245,117,16), -1)
        cv.putText(image, " ".join(sentence), (3,30),cv.FONT_HERSHEY_SIMPLEX, 1,(255,255,255), 2, cv.LINE_AA)
        cv.imshow("Feed", image)
        if cv.waitKey(10) & 0xFF ==ord("q"):
            break
            
                   
    cap.release()
    cv.destroyAllWindows()