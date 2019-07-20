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

def load_frame_into_numpy_array(image):
    (im_height,im_width) = image.shape[:2]
    return np.array(image).reshape((im_height, im_width, 3)).astype(np.uint8)




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
        if one_prediction['precision'] > 0.5 :
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
    image_np = load_frame_into_numpy_array(frame)
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    # Actual detection.
    output_dict = run_inference_for_single_image(image_np_expanded, default_object_detector)
    
    output = organize_object_prediction(output_dict,default_object_labels)
    
    return output



#############################################################
#############################################################

def draw_bounding_box_on_image_array(image,ymin,xmin,ymax,xmax,color='red',thickness=4,display_str_list=(),
                                     use_normalized_coordinates=True):
    """Adds a bounding box to an image (numpy array).
        Bounding box coordinates can be specified in either absolute (pixel) or
        normalized coordinates by setting the use_normalized_coordinates argument.
        Args:
        image: a numpy array with shape [height, width, 3].
        ymin: ymin of bounding box.
        xmin: xmin of bounding box.
        ymax: ymax of bounding box.
        xmax: xmax of bounding box.
        color: color to draw bounding box. Default is red.
        thickness: line thickness. Default value is 4.
        display_str_list: list of strings to display in box
        (each to be shown on its own line).
        use_normalized_coordinates: If True (default), treat coordinates
        ymin, xmin, ymax, xmax as relative to the image.  Otherwise treat
        coordinates as absolute.
        """
    image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
    draw_bounding_box_on_image(image_pil, ymin, xmin, ymax, xmax, color,
                                       thickness, display_str_list,
                                       use_normalized_coordinates)
    np.copyto(image, np.array(image_pil))
    return image

def draw_bounding_box_on_image(image,ymin,xmin,ymax,xmax,color='red',thickness=4,display_str_list=(),
                               use_normalized_coordinates=True):
    """Adds a bounding box to an image.
        Bounding box coordinates can be specified in either absolute (pixel) or
        normalized coordinates by setting the use_normalized_coordinates argument.
        Each string in display_str_list is displayed on a separate line above the
        bounding box in black text on a rectangle filled with the input 'color'.
        If the top of the bounding box extends to the edge of the image, the strings
        are displayed below the bounding box.
        Args:
        image: a PIL.Image object.
        ymin: ymin of bounding box.
        xmin: xmin of bounding box.
        ymax: ymax of bounding box.
        xmax: xmax of bounding box.
        color: color to draw bounding box. Default is red.
        thickness: line thickness. Default value is 4.
        display_str_list: list of strings to display in box
        (each to be shown on its own line).
        use_normalized_coordinates: If True (default), treat coordinates
        ymin, xmin, ymax, xmax as relative to the image.  Otherwise treat
        coordinates as absolute.
        """
    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size
    if use_normalized_coordinates:
        (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                            ymin * im_height, ymax * im_height)
    else:
        (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
    draw.line([(left, top), (left, bottom), (right, bottom),
                (right, top), (left, top)], width=thickness, fill=color)
    try:
        font = ImageFont.truetype('arial.ttf', 24)
    except IOError:
        font = ImageFont.load_default()
        
    # If the total height of the display strings added to the top of the bounding
    # box exceeds the top of the image, stack the strings below the bounding box
    # instead of above.
    display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
    # Each display_str has a top and bottom margin of 0.05x.
    total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)
            
    if top > total_display_str_height:
        #text_bottom = top
        text_bottom = bottom
    else:
        #text_bottom = bottom + total_display_str_height
        text_bottom = bottom
    # Reverse list and print from bottom to top.
    for display_str in display_str_list[::-1]:
        text_width, text_height = font.getsize(display_str)
        margin = np.ceil(0.05 * text_height)
        draw.rectangle(
                        [(left, text_bottom - text_height - 2 * margin), (left + text_width,
                                                                            text_bottom)],
                        fill=color)
        draw.text(
                    (left + margin, text_bottom - text_height - margin),
                    display_str,
                    fill='black',
                    font=font)
        text_bottom -= text_height - 2 * margin


def draw_bounding_boxes_on_image(image,boxes,color='red',thickness=4,display_str_list_list=()):
    """Draws bounding boxes on image.
        Args:
        image: a PIL.Image object.
        boxes: a 2 dimensional numpy array of [N, 4]: (ymin, xmin, ymax, xmax).
        The coordinates are in normalized format between [0, 1].
        color: color to draw bounding box. Default is red.
        thickness: line thickness. Default value is 4.
        display_str_list_list: list of list of strings.
        a list of strings for each bounding box.
        The reason to pass a list of strings for a
        bounding box is that it might contain
        multiple labels.
        Raises:
        ValueError: if boxes is not a [N, 4] array
        """
    boxes_shape = boxes.shape
    if not boxes_shape:
        return
    if len(boxes_shape) != 2 or boxes_shape[1] != 4:
        raise ValueError('Input must be of size [N, 4]')
    for i in range(boxes_shape[0]):
        display_str_list = ()
        if display_str_list_list:
            display_str_list = display_str_list_list[i]
        draw_bounding_box_on_image(image, boxes[i, 0], boxes[i, 1], boxes[i, 2],boxes[i, 3], color, thickness,display_str_list)

def draw_bounding_boxes_on_image_array(image,boxes,color='red',thickness=4,display_str_list_list=()):
    """Draws bounding boxes on image (numpy array).
        Args:
        image: a numpy array object.
        boxes: a 2 dimensional numpy array of [N, 4]: (ymin, xmin, ymax, xmax).
        The coordinates are in normalized format between [0, 1].
        color: color to draw bounding box. Default is red.
        thickness: line thickness. Default value is 4.
        display_str_list_list: list of list of strings.
        a list of strings for each bounding box.
        The reason to pass a list of strings for a
        bounding box is that it might contain
        multiple labels.
        Raises:
        ValueError: if boxes is not a [N, 4] array
        """
    image_pil = Image.fromarray(image)
    draw_bounding_boxes_on_image(image_pil, boxes, color, thickness,display_str_list_list)
    np.copyto(image, np.array(image_pil))
########################

def draw_object_boxes(frame,response):
    image_np = load_frame_into_numpy_array(frame)
    for r in response :
        image_np = draw_bounding_box_on_image_array(image_np,r['box'][0],r['box'][1],r['box'][2],r['box'][3],display_str_list=(r['category']))



def draw_object_boxes_image(image,response):
    
    image_np = load_image_into_numpy_array(image)

    for r in response :
        image_np = draw_bounding_box_on_image_array(image_np,r['box'][0],r['box'][1],r['box'][2],r['box'][3],display_str_list=([r['category']]))
    return image_np









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
        a = int(r["box"][0]*255)
        #print(a)
        b = int(r["box"][1]*255)
        #print(b)
        c = int(r["box"][2]*255)
        d = int(r["box"][3]*255)
        box = dlib.rectangle(a, b, c, d)
        top = box.top()
        right = box.right()
        bottom = box.bottom()
        left = box.left()
        cv2.rectangle(read_image, (left, top), (right, bottom), (0, 255, 0), 2)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(read_image, r["category"], (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (0, 255, 0), 2)
    return read_image




#def draw_object_boxes(read_image,response):
#   for (i,r) in enumerate(response):
        #print("\n \n the number ",i+1," prediction  is  :   ",r)
        #box = dlib.rectangle(r["box"][3], r["box"][0], r["box"][1], r["box"][2])
        #       top = r["box"][3]
        #        right = r["box"][0]
        #      bottom = r["box"][1]
        #      left = r["box"][2]
        #      cv2.rectangle(read_image, (left, top), (right, bottom), (255, 0, 0), 2)
        #     y = top - 15 if top - 15 > 15 else top + 15
        #     cv2.putText(read_image, r["category"], (left, y), cv2.FONT_HERSHEY_SIMPLEX,
        #               0.75, (255, 0, 0), 2)
# return read_image





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











def common_recognize_frame(frame):
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
        writer = cv2.VideoWriter(record_path, fourcc, 1,(frame.shape[1], frame.shape[0]), True)
    # loop over frames from the video file stream
    while True:
        # grab the frame from the threaded video stream
        frame = vs.read()
        response = common_recognize_frame(frame)
        #faces = recognize_frame(frame)
        #objects = recognize_objects_frame(frame)
        #print(objects)
        frame = draw_boxes(frame,response)
        #draw_object_boxes(frame,objects)
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
















