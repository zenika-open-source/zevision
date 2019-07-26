import sys
from imutils import paths
import cv2
from PIL import Image
sys.path.insert(0, '/home/ihab/ihabgit/zevision')

import lib.util as predict
imagePaths = list(paths.list_images("db_test"))

for image in imagePaths:
    
    response = predict.recognize_objects(image)
    print(response)
    print("\n\n\n\n")
#    test_image = Image.open(image)
    test_image = cv2.imread(image)
    test_image = predict.draw_object_boxes(test_image,response)
#    result_image = Image.fromarray(test_image)
#    result_image.save("results/"+image.split('/')[-1])
    predict.save_image(image,test_image,"results/")

