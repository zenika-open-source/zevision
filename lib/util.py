import imutils
import dlib
import cv2
import os
import numpy as np
import lib.models as models
import lib.encodings.encodings as codes
from imutils.video import VideoStream
import time
from PIL import Image
import PIL.ImageColor as ImageColor
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
import tensorflow as tf
from multiprocessing import Process, Queue

#recognize(encodings, boxes,data)
def predict_faces(encodings, boxes,data):
	response = []
# loop over the facial embeddings
	for (box,encoding) in zip(boxes,encodings):
		category = recognize_simple(encoding,data)
		prediction = {"category":category["category"],"precision":category["precision"],"box":box}
		response.append(prediction)
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


def save_image(image_path,drawn_image,path_to_results=""):
    image_name = image_path.split("/")[-1].split(".")[0]
    cv2.imwrite(path_to_results + image_name + ".jpg", drawn_image)

#recognize_camera (src=0,method="hog",encoding_path=default_path_encodings,record_path=None)
def launch_camera_feed (src=0,method="hog",encoding_path=default_path_encodings,record_path=None):
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
        #i += 1
        #if i == fps :
       #     i = 0
        #response = common_recognize_frame(frame)
        faces = recognize_faces_frame(frame)
        print(faces)
        print("\n\n\n")
        frame = draw_boxes(frame,faces)
# Do an iterator to make object detection work only once in multiple frames
        #if i == 0 : 
        start_time = time.time()
        objects = recognize_objects_frame(frame)
        print(objects)
        print(time.time() - start_time)
        frame = draw_object_boxes(frame,objects)
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
