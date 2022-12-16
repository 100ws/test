import face_recognition
import imutils
import pickle
import time
import cv2
from scipy.spatial import distance as dist
import numpy as np

import dlib


    
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
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

def glass_detection(img1,present,absent):

    #ref,img1=imcap.read()
    #clone=img.copy()
    #img1 = imcap.capture() # capture frame from video
    cv2.imwrite('temp.jpeg', img1)
    img = dlib.load_rgb_image('temp.jpeg')
    #plt.imshow(img)
    if len(detector(img))>0:
        rect = detector(img)[0]
        sp = predictor(img, rect)
        landmarks = np.array([[p.x, p.y] for p in sp.parts()])

        nose_bridge_x = []
        nose_bridge_y = []


        for i in [20,28,29,30,31,33,34,35]:
            nose_bridge_x.append(landmarks[i][0])
            nose_bridge_y.append(landmarks[i][1])


        ### x_min and x_max
        x_min = min(nose_bridge_x)
        x_max = max(nose_bridge_x)

        ### ymin (from top eyebrow coordinate),  ymax
        y_min = landmarks[20][1]
        y_max = landmarks[30][1]

        #img2=clone[x_min:x_max, y_min:y_max]
        img2 = Image.open('temp.jpeg')
        img2 = img2.crop((x_min,y_min,x_max,y_max))
        #plt.imshow(img2)
        
        img_blur = cv2.GaussianBlur(np.array(img2),(3,3), sigmaX=0, sigmaY=0)

        edges = cv2.Canny(image =img_blur, threshold1=100, threshold2=200)
        #plt.imshow(edges, cmap =plt.get_cmap('gray'))
        
        edges_center = edges.T[(int(len(edges.T)/2))]

        if 255 in edges_center:
            present=present+1
            print("Glasses are present")
        else:
                print("Glasses are absent")
                absent=absent+1
        return present,absent
def recog(img,currentname,encodingsP,data ):
    global EYE_AR_THRESH
    global MOUTH_AR_THRESH
    global LIP_DIST_THRESH
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
                    EYE_AR_THRESH=0.2
                    LIP_DIST_THRESH=50
                    MOUTH_AR_THRESH=0.8
            else:
                EYE_AR_THRESH = 0.2
                #EYE_AR_CONSEC_FRAMES = 15
                #MOUTH_AR_CONSEC_FRAMES = 10
                MOUTH_AR_THRESH=0.8
                LIP_DIST_THRESH=25



    return person



