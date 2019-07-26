from imutils import paths
import dlib

import cv2
import os
import numpy as np


import lib.models as models
import lib.encodings.encodings as codes





default_encoding_path = codes.default_encodings































# 2
# The default method currently is hog, which is the fastest one that doesn't take much resources.








# 4
def encode_training_faces(images):

	encodings=[]
	for image in images :

		encoding = np.array(models.face_encoder.compute_face_descriptor(image["raw"], image["landmark"],1))
		data = {"category":image["category"],"encoding":encoding}
		encodings.append(data)
	return encodings






















