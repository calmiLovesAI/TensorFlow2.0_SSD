from configuration import IoU_threshold, IMAGE_HEIGHT, IMAGE_WIDTH, background_sample_num
import numpy as np
from utils import IoU

class LabelAnchors():
    def __init__(self, anchors, labels, class_preds):
        # anchors : [batch_size, total number of anchors, 4]
        # labels shape :
        # [batch_size, num_of_objects_per_image(-1 for no object), 5(class + bbox coords)]
        # class_preds : [batch_size, total number of anchors, num_classes + 1]
        super(LabelAnchors, self).__init__()
        self.iou_threshold = IoU_threshold
        self.iou = IoU.IoU()
        self.anchors = anchors
        self.labels = labels
        self.class_preds = class_preds
        self.batch_size = labels.shape[0]
        self.num_anchors = anchors.shape[1]
        self.num_true_boxes = labels.shape[1]


    def __generate_iou_array(self, batch):
        iou_list = []
        for i in range(self.num_anchors):
            temp_list = []
            for j in range(self.num_true_boxes):
                iou_ij = self.iou.calculate_iou(coord_pred=self.anchors[0, i, :],
                                                coord_true=self.labels[batch, j, -4:])
                temp_list.append(iou_ij)
            iou_list.append(temp_list)

        return np.array(iou_list)  # shape: [self.num_anchors, self.num_true_boxes]


    def get_results(self):
        # Return : 3 arrays
        offset_list = []
        mask_list = []
        anchor_true_label_list = []
        for b in range(self.batch_size):
            offset_list_each = []
            mask_list_each = []
            anchor_true_label_list_each = []
            anchor_index_list = []
            # For every input image
            iou_array = self.__generate_iou_array(batch=b)
            iou_array_copy = iou_array
            while(iou_array.shape[1]):
                # First find the maximum value of all elements of the matrix, and then
                # according to the row and column where the maximum value is located,
                # we can obtain the corresponding anchor and true box match.
                match_list = self.__get_the_max_value_of_matix(iou_array)
                anchor_index = match_list[0]
                anchor_index_list.append(anchor_index)
                true_box_index = match_list[1]
                # Calculate the offset value.
                offset_array = self.__get_offset(pred_array=self.anchors[0, anchor_index, :],
                                                 true_array=self.labels[b, true_box_index, -4:])
                offset_list_each.append(offset_array)
                mask_list_each.append([1., 1., 1., 1.])
                anchor_true_label_list_each.append([self.labels[b, true_box_index, 0]])
                iou_array = self.__throw_row_and_col(iou_array, anchor_index, true_box_index)
            # All true boxes have been allocated, then we allocate the remaining anchors.
            for i in range(self.num_anchors):
                if i not in anchor_index_list:
                    # j : true box index
                    j = self.__get_max_value_of_one_row(i, iou_array_copy)
                    # calculate iou
                    iou_value = self.iou.calculate_iou(coord_pred=self.anchors[0, i, :],
                                                       coord_true=self.labels[b, j, -4:])
                    # The anchor is evaluated as a positive sample only if iou_value is greater than iou_threshold.
                    if iou_value > self.iou_threshold:
                        # calculate offset
                        offset_array = self.__get_offset(pred_array=self.anchors[0, i, :],
                                                         true_array=self.labels[b, j, -4:])
                        offset_list_each.append(offset_array)
                        mask_list_each.append([1., 1., 1., 1.])
                        anchor_true_label_list_each.append([self.labels[b, j, 0]])
                    else:
                        # the confidence that the anchor belongs to category 0 (background)
                        cls_0_confidence = self.class_preds[b, i, 0]
                        offset_list_each.append([0., 0., 0., 0.])
                        mask_list_each.append([0., 0., 0., 0.])
                        anchor_true_label_list_each.append([cls_0_confidence])

            offset_list.append(np.array(offset_list_each).flatten())
            mask_list.append(np.array(mask_list_each).flatten())
            anchor_true_label_list.append(self.__take_a_piece_from_array(np.array(anchor_true_label_list_each).flatten()))

        offset_list_array = np.array(offset_list)
        mask_list_array = np.array(mask_list)
        anchor_true_label_list_array = np.array(anchor_true_label_list)
        return offset_list_array, mask_list_array, anchor_true_label_list_array


    def __take_a_piece_from_array(self, array):
        temp = np.sort(array, axis=-1)
        n = background_sample_num
        while n :
            if temp[n] >= 1:
                n -= 1
            else:
                break

        t = temp[n]
        for i in range(array.shape[0]):
            if array[i] < t:
                array[i] = -1
            elif t <= array[i] < 1:
                array[i] = 0

        return array


    def __get_max_value_of_one_row(self, index, matrix):
        max_index = np.argmax(matrix, axis=1)
        return max_index[index]


    # Return the row and column where the maximum value is located.
    def __get_the_max_value_of_matix(self, matrix):
        max_index = np.argmax(matrix)
        i = max_index % matrix.shape[0]
        j = max_index % matrix.shape[1]
        return [i, j]

    def __get_offset(self, pred_array, true_array):
        pred_array = pred_array * [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_HEIGHT]
        offset_list = []
        for index in range(4):
            offset_list.append(true_array[index] - pred_array[index])

        offset_array = np.array(offset_list)

        return offset_array


    def __throw_row_and_col(self, matrix, i, j):
        # Throw away a row and a column of the matrix.
        n = matrix.shape[0]
        m = matrix.shape[1]
        part_1 = matrix[0: i, 0: j]
        part_2 = matrix[0: i, j + 1: m]
        part_3 = matrix[i + 1: n, 0: j]
        part_4 = matrix[i + 1: n, j + 1: m]

        return np.concatenate((np.concatenate((part_1, part_2), axis=1), np.concatenate((part_3, part_4), axis=1)), axis=0)



