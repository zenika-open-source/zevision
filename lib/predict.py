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
import tensorflow as tf


# Gotta create a proper pipeline for the data flow of object and face recognition !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

default_path_encodings = codes.default_encodings
default_encoding_data = codes.encoding_data

default_object_labels = models.object_labels
default_object_detector = models.inception_object_detector




def face_distance(face_encodings, face_to_compare):
	if len(face_encodings) == 0:
		return np.empty((0))

	return np.linalg.norm(face_encodings - face_to_compare)


def detection_method(method):
	if method == "cnn":
		face_detector = models.cnn_face_detector
	elif method == "haar":
		face_detector = models.haar_face_detector.detectMultiScale
	elif method == "hog":
		face_detector = models.hog_face_detector
	else :
		face_detector = None

	return face_detector





# Object Recognition stuff, cleanup for later !!!!
####################################################################################################################
####################################################################################################################
####################################################################################################################
####################################################################################################################
####################################################################################################################
####################################################################################################################


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

def get_labels(path_labels):
    indexes = [i for i in range(1,81)]
    f = open(path_labels, "r")
    labels = f.read()
    f.close()
    labels_ar = labels.split("\n")
    zipbObj = zip(indexes, labels_ar)
    dictOfLabels = dict(zipbObj)
    return dictOfLabels


def run_inference_for_single_image(image, graph):
    with graph.as_default():
        with tf.Session() as sess:
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                        'num_detections', 'detection_boxes', 'detection_scores',
                        'detection_classes', 'detection_masks'
                        ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
            if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(detection_masks, detection_boxes, image.shape[1], image.shape[2])
                detection_masks_reframed = tf.cast(tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(detection_masks_reframed, 0)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
            # Run inference
            output_dict = sess.run(tensor_dict,feed_dict={image_tensor: image})
        # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.int64)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict




def organize_object_prediction(output_dict,path_labels):
    labels = get_labels(path_labels)
    predictions = []
    for i in range(0,output_dict['num_detections']):
        index = output_dict['detection_classes'][i]
        category = labels[index]
        one_prediction = {"category" : category, "precision" : output_dict['detection_scores'][i],"box" : output_dict['detection_boxes'][i]}
        predictions.append(one_prediction)
    return predictions




def recognize_objects(image_path):
    image = Image.open(image_path)
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    image_np = load_image_into_numpy_array(image)
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    # Actual detection.
    output_dict = run_inference_for_single_image(image_np_expanded, default_object_detector)

    output = organize_object_prediction(output_dict,default_object_labels)
    return output



def recognize_objects_frame(frame):
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    #processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_np = load_image_into_numpy_array(frame)
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    # Actual detection.
    output_dict = run_inference_for_single_image(image_np_expanded, default_object_detector)
    
    output = organize_object_prediction(output_dict,default_object_labels)
    return output



                      
                      
                      
                      
                      
                      
                      
                      
                      
                      
                      
                      
                      
                      
                      
                      
                      
                      
                      
                      
                      
                      
                      
                      
                      
                      
                      
                      
                      
                      
                      
                      
                      
                      
                      
                      
                      
                      
                      
                      
                      
                      
                      
                      
                      
                      
                      
                      
                      
                      
                      
                      
                      
                      
                      
                      
                      
                      
                      
                      
                      
                      
                      
                      
                      
                      
                      
                      
                      
                      #End of object recognition
####################################################################################################################
####################################################################################################################
                      #Start of face recognition
                      ####################################################################################################################
                      ####################################################################################################################
                      ####################################################################################################################

#1
def preprocess(image,method="hog"):
	# load the input image and convert it from BGR to RGB

	img = cv2.imread(image)
	processed_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	return processed_image


# for frames in video, to process them without saving them
def preprocess_frame(frame,method="hog"):
    processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return processed_frame


#2
def detect_face_boxes_prediction(img,method="hog"):
	face_detector = detection_method(method)
	boxes = []

	raw_face_locations = face_detector(img, 1)

	for face in raw_face_locations :
		rect_to_css = face.top(), face.right(), face.bottom(), face.left() # this is just for HOG, do it for the other methods too
		boxes.append((max(rect_to_css[0], 0), min(rect_to_css[1], img.shape[1]), min(rect_to_css[2], img.shape[0]), max(rect_to_css[3], 0)))

	return boxes



#3
def detect_landmarks(processed_image,boxes):
	boxes = [dlib.rectangle(box[3], box[0], box[1], box[2]) for box in boxes]
	pose_predictor = models.pose_predictor_68_point
	raw_landmarks = [pose_predictor(processed_image, box) for box in boxes]
	return raw_landmarks

#4
def encode(processed_image,raw_landmarks):
	encodings = [np.array(models.face_encoder.compute_face_descriptor(processed_image, raw_landmark_set,1)) for raw_landmark_set in raw_landmarks]
	return encodings



def recognize_simple(encoding,datas):
	matches = []
	for data in datas["data"] :
		match = (face_distance(data["encoding"], encoding) <= 0.6)
		matches.append(match)
	name = "Unknown"
	precision = 1

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
			name = datas["data"][i]["category"]
			counts[name] = counts.get(name, 0) + 1

		# determine the recognized face with the largest number of
		# votes (note: in the event of an unlikely tie Python will
		# select first entry in the dictionary)
		name = max(counts, key=counts.get)
		precision = counts.get(name,0)/len(matchedIdxs)
	response = {"category" : name,"precision":precision}
	return response


#5
def recognize(encodings, boxes,data):
	response = []
# loop over the facial embeddings
	for (box,encoding) in zip(boxes,encodings):
		category = recognize_simple(encoding,data)
		prediction = {"category":category["category"],"precision":category["precision"],"box":box}
		response.append(prediction)
	return response


def draw_boxes(read_image,response):
    for (i,r) in enumerate(response):
        #print("\n \n the number ",i+1," prediction  is  :   ",r)
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

def save_image(image_path,drawn_image,path_to_results=""):
    image_name = image_path.split("/")[-1].split(".")[0]
    cv2.imwrite(path_to_results + image_name + ".jpg", drawn_image)


def recognize_frame(frame,method="hog",encoding_path=default_path_encodings):
    if encoding_path == codes.default_encodings :
        data = default_encoding_data
    else :
        data = codes.load_encodings(encoding_path)
    processed_frame = preprocess_frame(frame,method)
    boxes = detect_face_boxes_prediction(processed_frame,method)
    raw_landmarks = detect_landmarks(processed_frame,boxes)
    encodings = encode(processed_frame,raw_landmarks)
    response = recognize(encodings, boxes,data)
    return response


def recognize_face(image,method="hog",encoding_path=default_path_encodings):
	if encoding_path == codes.default_encodings :
		data = default_encoding_data
	else :
		data = codes.load_encodings(encoding_path)
	processed_image = preprocess(image,method)
	boxes = detect_face_boxes_prediction(processed_image,method)
	raw_landmarks = detect_landmarks(processed_image,boxes)
	encodings = encode(processed_image,raw_landmarks)
	response = recognize(encodings, boxes,data)
	return response











def common_recognize_frame (frame):
    faces = recognize_frame(frame)
    objects = recognize_objects_frame(frame)
    final_response = faces + objects
    return final_response







def recognize_camera (src=0,method="hog",encoding_path=default_path_encodings,record_path=None):
    # initialize the video stream, then allow the camera sensor to warm up
    print("[INFO] starting video stream...")
    vs = VideoStream(src).start()
    writer = None
    time.sleep(2.0)
    # start the FPS throughput estimator
    #fps = FPS().start()
    
    frame = vs.read()
    if record_path != None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(record_path, fourcc, 5,(frame.shape[1], frame.shape[0]), True)
    # loop over frames from the video file stream
    while True:
        # grab the frame from the threaded video stream
        frame = vs.read()
        response = common_recognize_frame(frame)
        draw_boxes(frame,response)
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
















