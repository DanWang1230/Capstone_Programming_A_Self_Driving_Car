from styx_msgs.msg import TrafficLight
import os
import tensorflow as tf
import numpy as np
import cv2

class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        curr_dir = os.path.dirname(os.path.realpath(__file__))     
        MODEL_PATH = curr_dir + '/sim_frozen_inference_graph.pb'
        
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            
            with tf.gfile.GFile(MODEL_PATH, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
            
            self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
            self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
            self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
            self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
            self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
        
        self.sess = tf.Session(graph=self.detection_graph)

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        # convert to rgb image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # expand dimension to input to the model
        image_expanded = np.expand_dims(image, axis=0)
        
        (boxes, scores, classes, num) = self.sess.run(
            [self.detection_boxes, self.detection_scores,
             self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_expanded})
        
        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        classes = np.squeeze(classes)
        
        cutoff_thresh = 0.7
        idx = []
        for i in range(len(classes)):
            if scores[i] >= cutoff_thresh:
                idx.append(i)
                
        new_boxes = boxes[idx, ...]
        new_scores = scores[idx, ...]
        new_classes = classes[idx, ...]
        
        if len(new_scores) > 0:
            class_number = int(new_classes[np.argmax(new_scores)]) # find the class with the largest score
        else:
            class_number = 4
            
        if class_number == 1:
            return TrafficLight.GREEN
        elif class_number == 2:
            return TrafficLight.YELLOW
        elif class_number == 3:
            return TrafficLight.RED
        else:
            return TrafficLight.UNKNOWN
        
        return TrafficLight.GREEN
