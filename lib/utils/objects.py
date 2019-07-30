from PIL import Image
import numpy as np
import tensorflow as tf
import lib.utils.models.models as models
import lib.utils.encodings.encodings as codes


default_object_labels = models.object_labels
default_object_detector = models.inception_object_detector



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
