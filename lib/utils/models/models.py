import cv2
import dlib
import os
import tensorflow as tf
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam
from keras.layers.pooling import MaxPooling2D

#def init():


def load_graph(frozen_graph_filename):
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(frozen_graph_filename, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return detection_graph

directory = os.path.dirname(os.path.abspath(__file__))


# Emotion detection model using keras
# Create the model
emotion_model = Sequential()

emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Flatten())
emotion_model.add(Dense(1024, activation='relu'))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(7, activation='softmax'))
emotion_model.load_weights(directory+'/emotion_model.h5')

global emotion_detector
emotion_detector = emotion_model
global emotion_dict
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

    # Hog method face detector
global hog_face_detector
hog_face_detector = dlib.get_frontal_face_detector()

# Object detection inception model
inception_path = directory + "/frozen_mobilenet_graph.pb"

inception_graph = load_graph(inception_path)
global inception_object_detector
inception_object_detector = inception_graph

# Object detection labels
label_path = directory + "/labels.txt"
global object_labels
object_labels = label_path


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
