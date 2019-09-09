import tensorflow as tf


class MultiBoxGenerator():
    def __init__(self, anchors, labels, class_preds):
        # labels shape :
        # [batch_size, num_of_objects_per_image(-1 for no object), 5(class + bbox coords)]
        super(MultiBoxGenerator, self).__init__()
        pass