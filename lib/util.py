import imutils
import dlib
import cv2
import os
import numpy as np
import lib.utils.models.models as models
import lib.utils.encodings.encodings as codes
import lib.utils.faces as face
import lib.utils.objects as obj
from imutils.video import VideoStream
import time
from PIL import Image
import tensorflow as tf


default_path_encodings = codes.default_encodings
default_encoding_data = codes.encoding_data

default_object_labels = models.object_labels
default_object_detector = models.inception_object_detector





def save_image(image_path,drawn_image,path_to_results=""):
    image_name = image_path.split("/")[-1].split(".")[0]
    cv2.imwrite(path_to_results + image_name + ".jpg", drawn_image)


def recognize_objects(image_path):
    image = Image.open(image_path)
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    image_np = obj.load_image_into_numpy_array(image)
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    # Actual detection.
    output_dict = obj.run_inference_for_single_image(image_np_expanded, default_object_detector)

    output = obj.organize_object_prediction(output_dict,default_object_labels)
    return output


def draw_object_boxes(read_image,response):
    height , width = read_image.shape[:2]
    for (i,r) in enumerate(response):
        box = dlib.rectangle(int(r["box"][1]*width), int(r["box"][0]*height), int(r["box"][3]*width), int(r["box"][2]*height))
        top = box.top()
        right = box.right()
        bottom = box.bottom()
        left = box.left()
        cv2.rectangle(read_image, (left, top), (right, bottom), (0, 255, 0), 2)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(read_image, r["category"], (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (0, 255, 0), 2)
    return read_image


# train_model(folder, method ="hog",encoding_path=default_encoding_path
def train_face_model(folder, method ="hog",encoding_path=default_path_encodings):
	processed_images = face.get_images(folder,method)
	detected_images = face.training_face_detection(processed_images,method)
	image_landmarks = face.detect_landmarks_training(detected_images)
	encodings = face.encode_training_faces(image_landmarks)
	# write the encodings in a file
	codes.write_encodings(encodings,encoding_path)
	codes.update()
	print("End of training")


#recognize_face(image,method="hog",encoding_path=default_path_encodings)
def predict_faces(image,method="hog",encoding_path=default_path_encodings):
	if encoding_path == codes.default_encodings :
		data = default_encoding_data
	else :
		data = codes.load_encodings(encoding_path)
	processed_image = face.preprocess(image,method)
	boxes = face.detect_face_boxes_prediction(processed_image,method)
    emotion_response = face.detect_emotions(processed_image,boxes)
	raw_landmarks = face.detect_landmarks_prediction(processed_image,boxes)
	encodings = face.encode_prediction(processed_image,raw_landmarks)
	recognition_response = face.recognize(encodings, boxes,data)
    response = face.match_face_emotion(recognition_response,emotion_response)
	return response



#draw_boxes(read_image,response)
def draw_face_boxes(read_image,response):
    for (i,r) in enumerate(response):
        box = dlib.rectangle(r["box"][3], r["box"][0], r["box"][1], r["box"][2])
        top = box.top()
        right = box.right()
        bottom = box.bottom()
        left = box.left()
        cv2.rectangle(read_image, (left, top), (right, bottom), (0, 255, 0), 2)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(read_image, r["category"], (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (0, 255, 0), 2)
    return read_image














#recognize_camera (src=0,method="hog",encoding_path=default_path_encodings,record_path=None)
def launch_camera_feed (src=0,method="hog",encoding_path=default_path_encodings,record_path=None,pred="all"):
    # initialize the video stream, then allow the camera sensor to warm up
    print("[INFO] starting video stream...")
    vs = VideoStream(src).start()
    writer = None
    time.sleep(2.0)
    # start the FPS throughput estimator
    #fps = FPS().start()
    fps = 1
    #iterator for the object detection to be activated
    #i = 0
    #q = Queue()
    frame = vs.read()
    if record_path != None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(record_path, fourcc, fps,(frame.shape[1], frame.shape[0]), True)
    # loop over frames from the video file stream
    while True:
        # grab the frame from the threaded video stream
        frame = vs.read()
        if pred == "all" :
            faces = face.recognize_faces_frame(frame)
            frame = draw_face_boxes(frame,faces)
            objects = obj.recognize_objects_frame(frame)
            frame = draw_object_boxes(frame,objects)
        elif pred == "obj" :
            objects = obj.recognize_objects_frame(frame)
            frame = draw_object_boxes(frame,objects)
        elif pred == "face" :
            faces = face.recognize_faces_frame(frame)
            frame = draw_face_boxes(frame,faces)
        else :
            print("please specify in launch_camera_feed the type of recognition you need. The arguments are 'obj' for objects, 'face' for faces, and 'all' for both")
            break
        cv2.imshow("Frame", frame)
        # Write the video in a the zevision/test/results/
        if record_path != None:
            writer.write(frame)
        key = cv2.waitKey(1) & 0xFF
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()
    # check to see if the video writer point needs to be released
    if writer is not None:
        writer.release()














## To be modified for usasge with API (tests still necessary) for both train and prediction with the function names adapted
if __name__ == '__main__':

    import argparse

    DATA_RAW = 'data/raw'
    METHOD = 'hog'
    ALLOWED_METHODS = ['hog', 'haar', 'cnn']

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-d', '--data', default=DATA_RAW,
        help='Path to data folder. Default: %s' % DATA_RAW)
    parser.add_argument('-m', '--method', default=METHOD,
        help='Method used by the model. One of: [%s]. Default: %s' % (', '.join(ALLOWED_METHODS), METHOD))
    args = parser.parse_args()

    assert args.method in ALLOWED_METHODS, \
        '--method should be one of: [%s]. %s given.' % (', '.join(ALLOWED_METHODS), args.method)

    train_face_model(args.data, args.method)
