import tensorflow as tf
import configuration

class Parse():
    def __init__(self):
        self.batch_size = configuration.BATCH_SIZE
        self.num_classes = configuration.NUM_CLASSES
        self.iou_threshold = configuration.IoU_threshold

    def get_batch_size(self):
        return self.batch_size

    def get_num_classes(self):
        return self.num_classes

    def get_iou_threshold(self):
        return self.iou_threshold