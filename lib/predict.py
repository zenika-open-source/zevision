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




default_path_encodings = codes.default_encodings
default_encoding_data = codes.encoding_data

default_object_labels = models.object_labels
default_object_detector = models.inception_object_detector




