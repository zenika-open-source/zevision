from imutils import paths
import dlib

import cv2
import os
import numpy as np


import lib.models as models
import lib.encodings.encodings as codes





default_encoding_path = codes.default_encodings






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




def detect_face_boxes(img,method="hog"):
	face_detector = detection_method(method)
	boxes = []

	raw_face_locations = face_detector(img["image"], 1)

	for face in raw_face_locations :
		rect_to_css = face.top(), face.right(), face.bottom(), face.left() # this is just for HOG, do it for the other methods too
		boxes.append((max(rect_to_css[0], 0), min(rect_to_css[1], img["image"].shape[1]), min(rect_to_css[2], img["image"].shape[0]), max(rect_to_css[3], 0)))

	return boxes








# 2
# The default method currently is hog, which is the fastest one that doesn't take much resources.

def training_face_detection(images,method="hog"):
	detected_images = []
	for image in images :
		boxes = detect_face_boxes(image,method)
		for box in boxes:
			detected_image = {"category" : image["category"], "raw":image["raw"],"box" : box}
			detected_images.append(detected_image)
	return detected_images



# 3
def detect_landmarks(images):
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











# Final used one
def train_model(folder, method ="hog",encoding_path=default_encoding_path):
	processed_images = get_images(folder,method)
	detected_images = training_face_detection(processed_images,method)
	image_landmarks = detect_landmarks(detected_images)
	encodings = encode_training_faces(image_landmarks)
	# write the encodings in a file
	codes.write_encodings(encodings,encoding_path)
	codes.update()
	print("End of training")









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

    train_model(args.data, args.method)
