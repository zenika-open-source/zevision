import sys
from imutils import paths
import cv2
import dlib
sys.path.insert(0, '/home/ihab/ihabgit/zevision')


import lib.util as predict
imagePaths = list(paths.list_images("db_test"))
for image in imagePaths:
    
    response = predict.predict_faces(image)
    print(response)
    test_image = cv2.imread(image)
    test_image = predict.draw_face_boxes(test_image,response)
    predict.save_image(image,test_image,"results/")
