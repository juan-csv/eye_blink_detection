import config as cfg
import dlib
import cv2
import numpy as np
from imutils import face_utils
from scipy.spatial import distance as dist



class eye_blink_detector():
    def __init__(self):
        # cargar modelo para detecction frontal de rostros
        self.detector_faces = dlib.get_frontal_face_detector()
        # cargar modelo para deteccion de puntos de ojos
        self.predictor_eyes = dlib.shape_predictor(cfg.eye_landmarks)

    def eye_blink(self,gray,rect,COUNTER,TOTAL):
        (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = self.predictor_eyes(gray, rect)
        shape = face_utils.shape_to_np(shape)
        # extract the left and right eye coordinates, then use the
        # coordinates to compute the eye aspect ratio for both eyes
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = self.eye_aspect_ratio(leftEye)
        rightEAR = self.eye_aspect_ratio(rightEye)
        # average the eye aspect ratio together for both eyes
        ear = (leftEAR + rightEAR) / 2.0
        # check to see if the eye aspect ratio is below the blink
        # threshold, and if so, increment the blink frame counter
        if ear < cfg.EYE_AR_THRESH:
            COUNTER += 1
        # otherwise, the eye aspect ratio is not below the blink
        # threshold
        else:
            # if the eyes were closed for a sufficient number of
            # then increment the total number of blinks
            if COUNTER >= cfg.EYE_AR_CONSEC_FRAMES:
                TOTAL += 1
            # reset the eye frame counter
            COUNTER = 0
        return COUNTER,TOTAL

    def eye_aspect_ratio(self,eye):
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





def convert_rectangles2array(rectangles,image):
    res = np.array([])
    for box in rectangles:
        [x0,y0,x1,y1] = max(0, box.left()), max(0, box.top()), min(box.right(), image.shape[1]), min(box.bottom(), image.shape[0])
        new_box = np.array([x0,y0,x1,y1])
        if res.size == 0:
            res = np.expand_dims(new_box,axis=0)
        else:
            res = np.vstack((res,new_box))
    return res

def get_areas(boxes):
    areas = []
    for box in boxes:
        x0,y0,x1,y1 = box
        area = (y1-y0)*(x1-x0)
        areas.append(area)
    return areas

def bounding_box(img,box,match_name=[]):
    for i in np.arange(len(box)):
        x0,y0,x1,y1 = box[i]
        img = cv2.rectangle(img,
                    (x0,y0),
                    (x1,y1),
                    (0,255,0),3);
        if not match_name:
            continue
        else:
            cv2.putText(img, match_name[i], (x0, y0-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
    return img