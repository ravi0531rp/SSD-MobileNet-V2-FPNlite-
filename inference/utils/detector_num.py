import tensorflow as tf
import numpy as np
import cv2
import io
import re
from object_detection.utils import label_map_util
from utils.Helpers import Helpers
from config import Config
import time

helper = Helpers()
config = Config()

class Detector():

    def __init__(self):
        self.doSetup = True
        self.MODEL_NAME = None
        self.PATH_TO_CKPT = None
        self.PATH_TO_LABELS = None
        self.NUM_CLASSES = None
        self.label_map = None
        self.categories = None
        self.category_index = None
        self.detection_graph = None
        self.confidence_threshold = None
        self.detectionBoxList = []
        self.detect_fn = None
        self.container_type = None
        self.match_count = 0

    def setup(self, container_type, PATH_TO_CKPT, PATH_TO_LABELS, NUM_CLASSES, confidence_threshold, MODEL_NAME= 'wheel_graph'):
        if self.doSetup:

            # # Model preparation
            self.container_type = container_type
            self.confidence_threshold = confidence_threshold
            self.MODEL_NAME = MODEL_NAME

            # Path to frozen detection graph. This is the actual model that is used for the object detection.
            self.PATH_TO_CKPT = PATH_TO_CKPT

            # List of the strings that is used to add correct label for each box.
            self.PATH_TO_LABELS = PATH_TO_LABELS
            self.NUM_CLASSES = NUM_CLASSES
            self.detect_fn = tf.saved_model.load(self.PATH_TO_CKPT)

            # Loading label map
            self.label_map = PATH_TO_LABELS
            self.doSetup = False
        else:
            pass

    # get detections from model
    def getdetection(self, input_tensor):
        detections = self.detect_fn(input_tensor)
        categories = self.categories = label_map_util.create_category_index_from_labelmap(self.label_map,
                                                                                          use_display_name=True)
        return detections, categories

    # get classes bounding boxes and corresponding confidence/score
    def get_classes_and_boxes(self, detections):
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                      for key, value in detections.items()}
        detections['num_detections'] = num_detections

        # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
        boxes = detections['detection_boxes']

        newDetections = []
        newBoxes = []
        scores = []
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
        for i in range(boxes.shape[0]):
            try:
                class_name = str(self.categories[detections['detection_classes'][i]]['name'])
                if detections['detection_scores'][i] >= self.confidence_threshold:
                    newDetections.append(class_name)
                    newBoxes.append(boxes[i])
                    scores.append(detections['detection_scores'][i])
            except:
                print("key error: "+ str(i))

        return newDetections, newBoxes, scores

    # combination logic for getting and processing detections
    def getInferenceResults(self, original_image):

        # change to RGB
        original_image_bgr = original_image.copy()
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        boundingBoxImage = original_image_bgr.copy()
        img_height, img_width, img_channels = original_image.shape

        input_tensor = np.expand_dims(original_image.copy(), 0)

        # get detections
        t1 = time.time()
        detections, categories = self.getdetection(input_tensor)

        newDetections, newbbox, scores = self.get_classes_and_boxes(detections)

        # draw the boxes and get the predicted text
        filtered_detections = newDetections
        t2 = time.time()

        filtered_boxes = newbbox
        filtered_scores = scores
        for i in range(len(filtered_detections)):
            ymin, xmin, ymax, xmax = filtered_boxes[i]
            ymin = (ymin * img_height).astype(int)
            ymax = (ymax * img_height).astype(int)
            xmin = (xmin * img_width).astype(int)
            xmax = (xmax * img_width).astype(int)
            height = ymax - ymin
            fontScale = 1 #min(img_width, img_height) / (200)
            if str(filtered_detections[i]) == 'parcel':
                colorPaint = (0, 0, 255)
            else:
                colorPaint = (255, 0, 0)
            cv2.rectangle(boundingBoxImage, (xmin, ymin),
                          (xmax, ymax), (0, 255, 0), 1)
            cv2.putText(boundingBoxImage, str(str(filtered_detections[i])+ "," + str(int(filtered_scores[i]*100))), (xmin, ymin),
                        cv2.FONT_HERSHEY_DUPLEX, fontScale, colorPaint, 1)

        return boundingBoxImage, filtered_detections, filtered_boxes , t2-t1
