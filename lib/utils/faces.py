import lib.utils.models.models as models
import lib.utils.encodings.encodings as codes
import lib.utils.classifier as classifier
import imutils
import dlib
import cv2
import os
import numpy as np
from imutils import paths





default_path_encodings = codes.default_encodings
default_encoding_data = codes.encoding_data

default_emotion_detector = models.emotion_detector
default_emotion_dict = models.emotion_dict

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




def process_raw_images(imagePaths,method="hog"):
	# loop over the image paths
	processedImages = []
	for (i, imagePath) in enumerate(imagePaths):
		# extract the person name from the image path
		print("[INFO] processing image {}/{}".format(i + 1,
			len(imagePaths)))
		name = imagePath.split(os.path.sep)[-2]

		# load the input image and convert it from BGR (OpenCV ordering)
		# to dlib ordering (RGB) or GRAY coloring for haar

		image = cv2.imread(imagePath)
		if method == "haar":
			processed_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		else:
			processed_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		data = {"category" : name, "raw":image,"image" : processed_image}
		processedImages.append(data)
	return processedImages

# 1
def get_images(folder,method="hog"):
	imagePaths = list(paths.list_images(folder))
	processedImages = process_raw_images(imagePaths,method)
	return processedImages


def detect_face_boxes_training(img,method="hog"):
	face_detector = detection_method(method)
	boxes = []

	raw_face_locations = face_detector(img["image"], 1)

	for face in raw_face_locations :
		rect_to_css = face.top(), face.right(), face.bottom(), face.left() # this is just for HOG, do it for the other methods too
		boxes.append((max(rect_to_css[0], 0), min(rect_to_css[1], img["image"].shape[1]), min(rect_to_css[2], img["image"].shape[0]), max(rect_to_css[3], 0)))

	return boxes


def training_face_detection(images,method="hog"):
	detected_images = []
	for image in images :
		boxes = detect_face_boxes_training(image,method)
		for box in boxes:
			detected_image = {"category" : image["category"], "raw":image["raw"],"box" : box}
			detected_images.append(detected_image)
	return detected_images


# 3
def detect_landmarks_training(images):
	image_landmarks = []
	for i in range(0,len(images)):
		images[i]["box"] = dlib.rectangle(images[i]["box"][3], images[i]["box"][0], images[i]["box"][1], images[i]["box"][2])
		images[i]["raw"] = cv2.cvtColor(images[i]["raw"], cv2.COLOR_BGR2RGB)
		pose_predictor = models.pose_predictor_68_point

		raw_landmark = pose_predictor(images[i]["raw"], images[i]["box"])
		data = {"category":images[i]["category"],"raw":images[i]["raw"],"landmark":raw_landmark}
		image_landmarks.append(data)
	return image_landmarks

# 4
def encode_training_faces(images):

	encodings=[]
	for image in images :

		encoding = np.array(models.face_encoder.compute_face_descriptor(image["raw"], image["landmark"],1))
		data = {"category":image["category"],"encoding":encoding}
		encodings.append(data)
	return encodings










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

def detect_emotions(processed_image,boxes):
    faces_emotions = []
    for (x, y, w, h) in faces:
        gray = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
        roi_gray = gray[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
        prediction = default_emotion_detector.predict(cropped_img)
        maxindex = int(np.argmax(prediction))
        emotion = default_emotion_dict[maxindex]
        faces_emotions.append(emotion)
    return faces_emotions



def match_face_emotion(recognition_response,emotion_response):
    response = []
    for emotion,recognition in emotion_response,recognition_response :
        res = {"category":recognition["category"],"precision":recognition["precision"],"box":recognition["box"],"emotion":emotion}
        response.append(res)
    return response

#3
def detect_landmarks_prediction(processed_image,boxes):
	boxes = [dlib.rectangle(box[3], box[0], box[1], box[2]) for box in boxes]
	pose_predictor = models.pose_predictor_68_point
	raw_landmarks = [pose_predictor(processed_image, box) for box in boxes]
	return raw_landmarks



#4
def encode_prediction(processed_image,raw_landmarks):
	encodings = [np.array(models.face_encoder.compute_face_descriptor(processed_image, raw_landmark_set,1)) for raw_landmark_set in raw_landmarks]
	return encodings


def _recognize_simple(encoding,datas):
	matches = []
	for data in datas["data"] :
		match = (classifier.face_distance(data["encoding"], encoding) <= 0.6)
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
		category = _recognize_simple(encoding,data)
		prediction = {"category":category["category"],"precision":category["precision"],"box":box}
		response.append(prediction)
	return response


def recognize_faces_frame(frame,method="hog",encoding_path=default_path_encodings):
    if encoding_path == codes.default_encodings :
        data = default_encoding_data
    else :
        data = codes.load_encodings(encoding_path)
    processed_frame = preprocess_frame(frame,method)
    boxes = detect_face_boxes_prediction(processed_frame,method)
    raw_landmarks = detect_landmarks_prediction(processed_frame,boxes)
    encodings = encode_prediction(processed_frame,raw_landmarks)
    response = recognize(encodings, boxes,data)
    return response
