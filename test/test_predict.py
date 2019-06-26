import sys
from imutils import paths
import cv2
import dlib
sys.path.insert(0, '/home/ihab/ihabgit/zevision')


import lib.predict as predict
imagePaths = list(paths.list_images("db_test"))
for image in imagePaths:

	response = predict.recognize_face(image)

	test_image = cv2.imread(image)
	test_image = draw_boxes(test_image,response)
    save_image(image,test_image,"results/")


