import tensorflow as tf
import parameters
import numpy as np
from utils import IoU

class LabelAnchors():
    def __init__(self, anchors, labels, class_preds):
        # anchors : [batch_size, total number pf anchors, 4]
        # labels shape :
        # [batch_size, num_of_objects_per_image(-1 for no object), 5(class + bbox coords)]
        # class_preds : [batch_size, total number pf anchors, num_classes + 1]
        super(LabelAnchors, self).__init__()
        parameter = parameters.Parse()
        self.iou_threshold = parameter.get_iou_threshold()
        self.iou = IoU.IoU()
        self.anchors = anchors
        self.labels = labels
        self.batch_size = anchors.shape[0]
        self.num_anchors = anchors.shape[1]
        self.num_true_boxes = labels.shape[1]


    def generate_iou_array(self):
        iou_list = []
        for i in range(self.num_anchors):
            temp_list = []
            for j in range(self.num_true_boxes):
                iou_ij = self.iou.calculate_iou(n_dims_coord_pred=self.anchors[:, i, :],
                                                n_dims_coord_true=self.labels[:, j, -4:])
                temp_list.append(iou_ij)
            iou_list.append(temp_list)

        return np.array(iou_list)


