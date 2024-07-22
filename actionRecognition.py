import os
import numpy as np
import sklearn.metrics
import tensorflow as tf
import cv2 as cv
import time
import mediapipe as mp
import sklearn
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

dpath="data"
actions=np.array(["thanks", "hello", "i love you"])
frame_seq=range(40)
seq_length=50

for action in actions:
    for x in frame_seq:
        try:
            os.makedirs(os.path.join(dpath,action, str(x)))
        except:
            pass

def mp_Detection(image, model):
    image=cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image.flags.writeable=False
    results=model.process(image)
    image.flags.writeable=True
    image=cv.cvtColor(image, cv.COLOR_RGB2BGR)
    return image, results

mp_holistic=mp.solutions.holistic
mp_drawings=mp.solutions.drawing_utils

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

cap=cv.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    for action in actions:
        for cap_Frame in frame_seq:
            for frame_num in range(seq_length):
                isCap, frame = cap.read()
                image, results=mp_Detection(frame, holistic)
                draw_landmark(image, results)
                if frame_num==0:
                    cv.putText(image, "basliyor", (120,200),cv.FONT_HERSHEY_SIMPLEX, 1,(0,255,0), 4, cv.LINE_AA)
                    cv.putText(image, "toplaniyor {} numara {}".format(action,cap_Frame), (15,12),cv.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,255), 4, cv.LINE_AA)
                    cv.imshow("Feed", image)
                    cv.waitKey(2000)
                else:
                    cv.putText(image, "toplaniyor {} numara {}".format(action,cap_Frame), (15,12),cv.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,255), 4, cv.LINE_AA)
                    cv.imshow("Feed", image)
                keypoints=extract(results)
                npy_path=os.path.join(dpath, action,str(cap_Frame), str(frame_num))
                np.save(npy_path, keypoints)
                if cv.waitKey(10) & 0xFF ==ord("q"):
                    break
            
                   
    cap.release()
    cv.destroyAllWindows()

sequences=[]
labels=[]
label_map=dict([(y,x) for x , y in enumerate(actions)])
for action in actions:
    for seq in frame_seq:
        window=[]
        for frame_num in range(seq_length):
            res=np.load(os.path.join(dpath, action, str(seq), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])
x=np.array(sequences)
y=tf.keras.utils.to_categorical(labels).astype(int)
x_train, x_test, y_train, y_test =train_test_split(x, y, test_size=0.05)

model=tf.keras.models.Sequential([
    tf.keras.layers.LSTM(64,return_sequences=True, input_shape=(30,1662)),
    tf.keras.layers.LSTM(128,return_sequences=True),
    tf.keras.layers.LSTM(64,return_sequences=False),
    tf.keras.layerS.Dense(64,activation="relu"),
    tf.keras.layerS.Dense(64,activation="relu"),
    tf.keras.layerS.Dense(actions.shape[0],activation="softmax")
])

model.compile(optimizer="Adam", loss="categorical_crossentropy", metric=["categorical_accuracy"])
model.fit(x_train, y_train, epochs=1500)

model.save("model.hs")

yhat=model.predict(x_test)
ytrue=np.argmax(y_test, axis=1).to_list()
yhat=np.argmax(yhat, axis=1).to_list()

conf_matrix=sklearn.metrics.multilabel_confusion_matrix(ytrue,yhat)
acc=sklearn.metics.accuracy_score(ytrue, yhat)