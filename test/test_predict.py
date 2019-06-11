import sys
from imutils import paths
import cv2
import dlib
sys.path.insert(0, '/home/bendidiihab/ihabgit/zihub')


import lib.predict as predict
imagePaths = list(paths.list_images("db_test"))
for image in imagePaths:

	response = predict.recognize_face(image)

	test_image = cv2.imread(image)
	for (i,r) in enumerate(response):
		print("\n \n the number ",i+1," prediction for ",image," is  :   ",r)
		box = dlib.rectangle(r["box"][3], r["box"][0], r["box"][1], r["box"][2])
		top = box.top()
		right = box.right()
		bottom = box.bottom()
		left = box.left()
		cv2.rectangle(test_image, (left, top), (right, bottom), (0, 255, 0), 2)
		y = top - 15 if top - 15 > 15 else top + 15
		cv2.putText(test_image, r["category"], (left, y), cv2.FONT_HERSHEY_SIMPLEX,
			0.75, (0, 255, 0), 2)

	image = image.split("/")[-1].split(".")[0]
	cv2.imwrite("results/" + image + ".jpg", test_image)
