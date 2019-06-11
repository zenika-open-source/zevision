import cv2
import dlib
import os

#def init():

directory = os.path.dirname(os.path.abspath(__file__))

    # Hog method face detector
global hog_face_detector
hog_face_detector = dlib.get_frontal_face_detector()

    	# Haar Cascades method face detector
global haar_face_detector
haar_face_detector= cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    	# Face detector with CNN, needs GPU
cnn_face_detection_model = directory + "/mmod_human_face_detector.dat"
global cnn_face_detector
cnn_face_detector = dlib.cnn_face_detection_model_v1(cnn_face_detection_model)

    	# Face landmarks predictor model
predictor_68_point_model =  directory + "/shape_predictor_68_face_landmarks.dat"
global pose_predictor_68_point
pose_predictor_68_point = dlib.shape_predictor(predictor_68_point_model)

    	# face encoding model
face_recognition_model = directory + "/dlib_face_recognition_resnet_model_v1.dat"
global face_encoder
face_encoder = dlib.face_recognition_model_v1(face_recognition_model)
