import tensorflow as tf
from configuration import NUM_CLASSES
from core.ssd import ssd_prediction


class InferenceProcedure(object):
    def __init__(self, model):
        self.model = model
        self.num_classes = NUM_CLASSES + 1

    def __get_ssd_prediction(self, image):
        output = self.model(image, training=False)
        pred = ssd_prediction(feature_maps=output, num_classes=self.num_classes)
        return pred

    def get_final_boxes(self, image):
        pred_boxes = self.__get_ssd_prediction(image)
        # print(pred_boxes.shape)