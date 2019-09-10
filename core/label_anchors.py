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


    def __generate_iou_array(self):
        iou_list = []
        for i in range(self.num_anchors):
            temp_list = []
            for j in range(self.num_true_boxes):
                iou_ij = self.iou.calculate_iou(n_dims_coord_pred=self.anchors[:, i, :],
                                                n_dims_coord_true=self.labels[:, j, -4:])
                temp_list.append(iou_ij)
            iou_list.append(temp_list)

        return np.array(iou_list)  # shape: [self.num_anchors, self.num_true_boxes, batch_size, 1]

    def __get_the_max_value_of_matix(self, matrix):
        max_index = np.argmax(matrix)
        i = max_index % matrix.shape[0]
        j = max_index % matrix.shape[1]
        return [i, j]

    def __get_the_max_value_of_iou_array(self, iou_array):
        # iou_array = iou_array.reshape((self.num_anchors, self.num_true_boxes, -1))
        max_value_list = []
        for index in range(iou_array.shape[2]):
            max_list = self.__get_the_max_value_of_matix(iou_array[:, :, index, 1])
            max_value_list.append(max_list)

        return max_value_list

    def __get_offset(self, pred_array, true_array):
        offset_list = []
        for index in range(4):
            offset_list.append(true_array[index] - pred_array[index])

        offset_array = np.array(offset_list)

        return offset_array

    def get_results(self):
        # max_values : [[i1, j1], [i2, j2], ..., [in, jn]]
        max_values = self.__get_the_max_value_of_iou_array(iou_array=self.__generate_iou_array())

        for item in max_values:
            # for each picture entered


