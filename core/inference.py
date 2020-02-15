import tensorflow as tf
import numpy as np
from configuration import NUM_CLASSES, IMAGE_HEIGHT, IMAGE_WIDTH
from core.ssd import ssd_prediction
from utils.nms import NMS


class InferenceProcedure(object):
    def __init__(self, model):
        self.model = model
        self.num_classes = NUM_CLASSES + 1
        self.image_size = np.array([IMAGE_HEIGHT, IMAGE_WIDTH], dtype=np.float32)
        self.nms_op = NMS()

    def __get_ssd_prediction(self, image):
        output = self.model(image, training=False)
        pred = ssd_prediction(feature_maps=output, num_classes=self.num_classes)
        return pred

    @staticmethod
    def __resize_boxes(boxes, image_height, image_width):
        cx = boxes[..., 0] * image_width
        cy = boxes[..., 1] * image_height
        w = boxes[..., 2] * image_width
        h = boxes[..., 3] * image_height
        xmin = cx - w / 2
        ymin = cy - h / 2
        xmax = cx + w / 2
        ymax = cy + h / 2
        resized_boxes = tf.stack(values=[xmin, ymin, xmax, ymax], axis=-1)
        return resized_boxes

    def __filter_background_boxes(self, ssd_predict_boxes):
        num_of_total_predict_boxes = ssd_predict_boxes.shape[1]
        scores = tf.argmax(input=tf.nn.softmax(ssd_predict_boxes[..., :self.num_classes]), axis=-1)
        filtered_boxes_list = []
        for i in range(num_of_total_predict_boxes):
            if scores[:, i] != 0:
                filtered_boxes_list.append(ssd_predict_boxes[:, i, :])
        filtered_boxes = tf.stack(values=filtered_boxes_list, axis=1)
        return filtered_boxes

    def get_final_boxes(self, image):
        pred_boxes = self.__get_ssd_prediction(image)
        pred_boxes = self.__filter_background_boxes(pred_boxes)
        pred_boxes_class = tf.nn.softmax(logits=pred_boxes[..., :self.num_classes])
        pred_boxes_class = tf.reshape(tensor=pred_boxes_class, shape=(-1, self.num_classes))
        pred_boxes_coord = tf.math.sigmoid(pred_boxes[..., self.num_classes:])
        pred_boxes_coord = tf.reshape(tensor=pred_boxes_coord, shape=(-1, 4))
        resized_pred_boxes = self.__resize_boxes(boxes=pred_boxes_coord,
                                                 image_height=image.shape[1],
                                                 image_width=image.shape[2])
        box_tensor, score_tensor, class_tensor = self.nms_op.nms(boxes=resized_pred_boxes, box_scores=pred_boxes_class)
        return box_tensor, score_tensor, class_tensor

