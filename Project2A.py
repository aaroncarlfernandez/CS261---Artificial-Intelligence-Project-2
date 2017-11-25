#!/usr/bin/env python
# Project 2 on CS261 - Artificial Intelligence
# Author: Aaron Carl Fernandez
# This program aims to use Microsoft Cognitive Services Emotion API and Face API as a requirement for the project
# The game is scored for every correct emotion and gender combination. Game continues as player scores more than 2 points out of 3 tries. Perceived age is shown for every incorrect face.
#
import cv2
import sys
import numpy
import requests
import time
import operator
import math
import random
import cognitive_face as CF
import json

emotions = {
    "neutral": "neutral",
    "happiness": "happy",
    "contempt": "contempt",
    "sadness": "sad",
    "disgust": "disgusted",
    "anger": "angry",
    "surprise": "surprised",
    "fear": "surprised" }

genders = {
    "female": "female",
    "male": "male"
}

target_emotion = ""
target_gender = ""
faces = []
fapi_faces = []
processed = False

def send_pic(img):
    global emotion
    global faces
    global fapi_faces
    global processed

    img_str = cv2.imencode('.jpg', img)[1].tostring()
    data = img_str
    res = requests.post(url='https://api.projectoxford.ai/emotion/v1.0/recognize',
                        data=data,
                        headers={'Content-Type': 'application/octet-stream', 'Ocp-Apim-Subscription-Key': '469ca1e004274adf9af387a3a0813b4a'})

    res2 = requests.post(url='https://api.projectoxford.ai/face/v1.0/detect?returnFaceId=false&returnFaceLandmarks=false&returnFaceAttributes=age,gender',
                        data=data,
                        headers={'Content-Type': 'application/octet-stream', 'Ocp-Apim-Subscription-Key': '0082562a3f7c4dcd8da0cfb91d6328c9'})

    print("Emotion API response: " + str(res.status_code))
    if res.status_code == 200:
        faces = res.json()
        print("Face API response: " + str(res2.status_code))
        if res2.status_code == 200:
            fapi_faces = res2.json()
    processed = True


cascPath = sys.argv[1]
faceCascade = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture(0)
pic_width = int(video_capture.get(3))

lastTime = time.time()
interval = 5
timeCount= 0
target_emotion = emotions[random.choice(list(emotions)[:-1])]
target_gender = random.choice(list(genders))
game_over = False
game_level = 0

while game_over == False:
    game_score = 0
    game_level += 1
    for x in range(3):
        faces = []
        fapi_faces = []
        target_emotion = emotions[random.choice(list(emotions)[:-1])]
        target_gender = random.choice(list(genders))

        lastTime = time.time()
        score_bool = False
        while timeCount <= interval:
            ret, frame = video_capture.read()

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )

            fapi_faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )

            cv2.rectangle(frame, (0, 0), (pic_width, 50), (255, 255, 255, 128), -1)
            cv2.putText(frame, str(int(math.ceil(interval-timeCount))), (pic_width-50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2, cv2.LINE_AA)
        #OC
            if target_emotion == 'angry':
                cv2.putText(frame, "Show me an " + target_emotion.upper() + " " + target_gender.upper() + " face...", (0, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 0, 0), 2, cv2.LINE_AA)
            else:
                cv2.putText(frame, "Show me a " + target_emotion.upper() + " " + target_gender.upper() + " face...", (0, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, "Level: " + str(game_level), (pic_width - 200, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7 , (0, 0, 0), 2, cv2.LINE_AA)

            cv2.imshow('Video', frame)

            timeCount = time.time() - lastTime

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.rectangle(frame, (0, 0), (pic_width, 50), (255, 255, 255, 128), -1)
        cv2.putText(frame, "Analyzing...", (0, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        send_pic(frame)
        timeCount = 0
        lastTime = time.time()

        ret, frame = video_capture.read()
        cv2.rectangle(frame, (0, 0), (pic_width, 50), (255, 255, 255, 128), -1)
        cv2.putText(frame, "Analyzing..", (0, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow('Video', frame)

        while not processed:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        processed = False

        if len(faces) > 0:
            correct = 0
            for face in faces:
               emotion = emotions[max(face['scores'].items(), key=operator.itemgetter(1))[0]]
               left = int(face['faceRectangle']['left'])
               top = int(face['faceRectangle']['top'])
               width = int(face['faceRectangle']['width'])
               height = int(face['faceRectangle']['height'])

               color = (0, 0, 255)
               for fapi_face in fapi_faces:
                    gender = list(fapi_face['faceAttributes'].values())[0]
                    age = list(fapi_face['faceAttributes'].values())[1]

                    if (emotion == target_emotion and gender == target_gender):
                        color = (0, 255, 0)
                        score_bool = True
                        cv2.rectangle(frame, (left, top), (left + width, top + height), color, 2)
                        cv2.rectangle(frame, (left, top), (left + width, top - 40), color, -1)
                        cv2.putText(frame, emotion.title() + " " + gender.title(), (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1,cv2.LINE_AA)
                    else:
                        cv2.rectangle(frame, (left, top), (left + width + 70, top + height), color, 2)
                        cv2.rectangle(frame, (left, top), (left + width + 70, top - 40), color, -1)
                        cv2.putText(frame, emotion.title() + " " + gender.title() + " " + str(age) + " yrs old", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1,cv2.LINE_AA)

            cv2.rectangle(frame, (0, 0), (pic_width, 50), (255, 255, 255, 128), -1)

            if score_bool == True:
                game_score += 1
                if (x == 2 and game_score < 2):
                    cv2.putText(frame, "Game Over !!", (0, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
                else:
                    cv2.putText(frame, "You scored 1 point !!", (0, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
            else:
                if (x == 2 and game_score < 2):
                    cv2.putText(frame, "Game Over !!", (0, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
                else :
                    cv2.putText(frame, "Wrong face :(", (0, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, "Score: " + str(game_score), (pic_width-200, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

            cv2.imshow('Video', frame)
            if cv2.waitKey(3000) & 0xFF == ord('q'):
                break

    if game_score < 2:
        game_over = True
        break

# Release the capture
video_capture.release()
cv2.destroyAllWindows()