from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
#import playsound
import argparse
import imutils
import time
import dlib
import cv2
#import schedule
import time
from picamera import PiCamera
#from picamera.array import PiRGBArray
from PIL import Image
from imutils.video import VideoStream
from imutils.video import FPS
import face_recognition
import imutils
import pickle
import time
import cv2
from datetime import datetime

import RPi.GPIO as GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
BUZZER= 23
GPIO.setup(BUZZER, GPIO.OUT)



currentname = "unknown"
EYE_AR_THRESH = 0.2
MOUTH_AR_THRESH=0.8
LIP_DIST_THRESH=25
EYE_AR_CONSEC_FRAMES = 4
global COUNTER
global COUNTER1
global recog_frame
global continue_frame
global time_frame
global yes
global no
COUNTER=0
COUNTER1=0
yes=0
no=0
yes1=0
no1=0
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mouthstart,mouthEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

FACE_RECOG_FRAME=0
continue_frame=0
recog_frame=0
time_frame=0
encodingsP = "/home/pi/Desktop/final_drowsiness/encodings.pickle"
data = pickle.loads(open(encodingsP, "rb").read())

def eye_aspect_ratio(eye):
	# compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])
	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
	C = dist.euclidean(eye[0], eye[3])
	# compute the eye aspect ratio
	ear = (A + B) / (2.0 * C)
	# return the eye aspect ratio
	return ear

def mouth_aspect_ratio(mouth):
    A=dist.euclidean(mouth[2],mouth[10])
    B=dist.euclidean(mouth[3],mouth[9])
    C=dist.euclidean(mouth[4],mouth[8])
    D=dist.euclidean(mouth[0],mouth[6])
    mar=(A+B+C)/(3*D)
    return mar


def lip_distance(shape):
    top_lip = shape[50:53]
    top_lip = np.concatenate((top_lip, shape[61:64]))

    low_lip = shape[56:59]
    low_lip = np.concatenate((low_lip, shape[65:68]))

    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)

    distance = abs(top_mean[1] - low_mean[1])
    return distance

print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("/home/pi/Desktop/final_drowsiness/shape_predictor_68_face_landmarks.dat")
print("[INFO] starting video stream thread...")
#cap=VideoStream(usePicam=True).start()
#cap=cv2.VideoCapture(0)
fps = FPS().start()

def recog(img,currentname,encodingsP,data ):
    
    person=0
    #gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    boxes = face_recognition.face_locations(img)
    # compute the facial embeddings for each face bounding box
    encodings = face_recognition.face_encodings(img, boxes)
    names = []
    for encoding in encodings:
        # attempt to match each face in the input image to our known
        # encodings
        matches = face_recognition.compare_faces(data["encodings"],
            encoding)
        name = "Unknown" #if face is not recognized, then print Unknown

        # check to see if we have found a match
        if True in matches:
            # find the indexes of all matched faces then initialize a
            # dictionary to count the total number of times each face
            # was matched
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}

            # loop over the matched indexes and maintain a count for
            # each recognized face face
            for i in matchedIdxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1

            # determine the recognized face with the largest number
            # of votes (note: in the event of an unlikely tie Python
            # will select first entry in the dictionary)
            name = max(counts, key=counts.get)

            #If someone in your dataset is identified, print their name on the screen
            if currentname != name:
                currentname = name
                print(currentname)
                if currentname=="pragi":
                    person=1
                   
            else:
                #EYE_AR_THRESH = 0.2
                #EYE_AR_CONSEC_FRAMES = 15
                #MOUTH_AR_CONSEC_FRAMES = 10
                #MOUTH_AR_THRESH=0.8
                #LIP_DIST_THRESH=25
                print("no")



    return person,currentname
            


    
def printfaceid(name,frame,yes,no):
    
    if yes>no:
            cv2.putText(frame, name, (80, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    if no>yes:
        cv2.putText(frame, "Wrong ID", (80, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    


    

def processing(frame,encodingsP,data):

    # assigning global all variables 
    global name
    global COUNTER1
    global COUNTER
    global continue_frame
    global recog_frame
    global time_frame
    global yes
    global no

    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    rects=detector(gray,0)
    # detection for face present or not 
    if len(rects)==0:
        cv2.putText(frame, "Sit Properly Your face is not Visible", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    #processing on detected faces
    for rect in rects:
        
        continue_frame=continue_frame+1
        recog_frame=recog_frame+1
        # face detection at initial

        if recog_frame<10 :
            
            m,name=recog(frame,currentname,encodingsP,data)
            
            if m==1:
                yes=yes+1
            else:
                no=no+1

        #face detection at continous frame after 50 frames for 10 duration of frames

        if continue_frame>50:
            time_frame=time_frame+1
            yes=0
            no=0
            
            if time_frame<10:
                 m,name=recog(frame,currentname,encodingsP,data)
                 if m==1:
                     yes=yes+1
                 else:
                     no=no+1
            time_frame=0
            continue_frame=0

        # showing results
        printfaceid(name,frame,yes,no)
        
        #loading face and eye predictor

        shape=predictor(gray,rect)
        shape = face_utils.shape_to_np(shape)
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        mouth=shape[mouthstart:mouthEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        #EYE aspect ratio
        ear = (leftEAR + rightEAR) / 2.0
        #mouth aspect ratio
        mar=mouth_aspect_ratio(mouth)
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        mouthHull=cv2.convexHull(mouth)

        #lips distance
        distance=lip_distance(shape)

        # showing all aspect ratio

        cv2.putText(frame, "DIST: {:.2f}".format(distance), (300, 70),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "MAR: {:.2f}".format(mar), (300, 50),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        #eyes drowsiness alert
        if ear < EYE_AR_THRESH:
            COUNTER +=1
            #print("counter",COUNTER)
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                print("yes")
                
                time1=datetime.now()
                cv2.imwrite("drowsyfaces/newimage{}.jpg".format(time1),frame)
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                GPIO.output(BUZZER, GPIO.HIGH)
                time.sleep(2)
                GPIO.output(BUZZER, GPIO.LOW)
                #GPIO.cleanup()
        if ear>EYE_AR_THRESH:
            COUNTER=0
            GPIO.output(BUZZER, GPIO.LOW)
        
        #mouth alert
        if mar>= MOUTH_AR_THRESH and distance>LIP_DIST_THRESH:
            COUNTER1 +=1
            #cv2.putText(frame, " MOUTH  ALERT!", (10, 100),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if COUNTER1 >= 8:
                print("yes")
                #yawning +=1
                cv2.putText(frame, " MOUTH  ALERT!", (10, 50),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
def main():
    while True:
        #t = time.time()
        frame=cap.read()
        frame=imutils.resize(frame,width=480)
        processing(frame,encodingsP,data)
        fps.update()
        cv2.imshow("Frame",frame)
        key = cv2.waitKey(1) & 0xFF
       
        #print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

        if key == ord("q"):
            break
    fps.stop()
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    #print("FPS",fps.fps())
    cv2.destroyAllWindows()
    cap.stop()


