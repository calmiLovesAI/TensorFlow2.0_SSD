import tensorflow as tf
import configuration

class Parse():
    def __init__(self):
        self.batch_size = configuration.BATCH_SIZE
        self.num_classes = configuration.NUM_CLASSES

    def get_batch_size(self):
        return self.batch_size

    def get_num_classes(self):
        return self.num_classes
