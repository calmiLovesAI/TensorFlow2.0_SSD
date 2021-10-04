import numpy as np
import tensorflow as tf

from utils.tf_addons import clip_by_value


def intersect(box_a, box_b):
    """

    :param box_a: Tensor, shape: (A, 4)
    :param box_b: Tensor, shape: (B, 4)
    :return: box_a和box_b的交集的面积, shape: (A, B)
    """
    max_xy = tf.math.minimum(tf.expand_dims(box_a[:, 2:], axis=1),
                             tf.expand_dims(box_b[:, 2:], axis=0))
    min_xy = tf.math.minimum(tf.expand_dims(box_a[:, :2], axis=1),
                             tf.expand_dims(box_b[:, :2], axis=0))

    inter = clip_by_value(t=(max_xy - min_xy), clip_value_min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(box_a, box_b):
    """
    计算box_a和box_b的交并比(即IoU)
    :param box_a: Tensor, GT bounding boxes, shape: (a, 4)
    :param box_b: Tensor, 预测的bounding boxes, shape: (b, 4)
    :return: Tensor, shape: (a, b)
    """
    inter = intersect(box_a, box_b)
    area_a = (box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])
    area_a = tf.expand_dims(area_a, axis=1)
    area_b = (box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1])
    area_b = tf.expand_dims(area_b, axis=0)
    union = area_a + area_b - inter
    return inter / union


def match(threshold, truths, priors, variances, labels, loc_t, conf_t):
    """
    匹配真实框与DefaultBoxes
    :param threshold:
    :param truths:
    :param priors:
    :param variances:
    :param labels:
    :param loc_t:
    :param conf_t:
    :param idx:
    :return:
    """
    overlaps = jaccard(truths, point_form(priors))

    best_prior_overlap = tf.math.reduce_max(overlaps, axis=1, keepdims=True)
    best_prior_idx = tf.math.argmax(overlaps, axis=1)
    best_truth_overlap = tf.math.reduce_max(overlaps, axis=0, keepdims=True)
    best_truth_idx = tf.math.argmax(overlaps, axis=0)

    best_truth_overlap = tf.tensor_scatter_nd_update(tensor=tf.squeeze(best_truth_overlap),
                                                     indices=tf.expand_dims(best_prior_idx, axis=1),
                                                     updates=tf.fill(dims=[best_prior_idx.shape[0]], value=2.0))
    best_truth_overlap = tf.expand_dims(best_truth_overlap, axis=0)

    best_truth_idx = tf.tensor_scatter_nd_update(tensor=best_truth_idx,
                                                 indices=tf.expand_dims(best_prior_idx, axis=1),
                                                 updates=best_prior_idx)

    matches = tf.gather(params=truths, indices=best_truth_idx)
    conf = tf.gather(params=labels, indices=best_truth_idx) + 1
    conf = tf.where(condition=best_truth_overlap < threshold, x=0, y=conf)
    conf = tf.squeeze(conf)
    loc = encode(matches, priors, variances)
    loc_t.append(loc)
    conf_t.append(conf)


def encode(matched, priors, variances):
    g_cxcy = (matched[:, :2] + matched[:, 2:])/2 - priors[:, :2]
    g_cxcy /= (variances[0] * priors[:, 2:])
    g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
    g_wh = tf.math.log(g_wh) / variances[1]
    return tf.concat(values=[g_cxcy, g_wh], axis=1)



def point_form(boxes):
    """
    将boxes的坐标从(cx,cy, w, h)转换为(xmin, ymin, xmax, ymax)格式
    :param boxes:
    :return:
    """
    return tf.concat(values=[
        boxes[:, :2] - boxes[:, 2:] / 2, boxes[:, :2] + boxes[:, 2:] / 2
    ], axis=1)


def center_size(boxes):
    """
    将boxes的坐标从(xmin, ymin, xmax, ymax)转换为(cx,cy, w, h)格式
    :param boxes:
    :return:
    """
    return tf.concat(values=[
        (boxes[:, 2:] + boxes[:, :2]) / 2, boxes[:, 2:] - boxes[:, :2]
    ], axis=1)


# # The last dimensions of box_1 and box_2 are both 4. (x, y, w, h)
# class IOU(object):
#     def __init__(self, box_1, box_2):
#         self.box_1_min, self.box_1_max = self.__get_box_min_and_max(box_1)
#         self.box_2_min, self.box_2_max = self.__get_box_min_and_max(box_2)
#         self.box_1_area = self.__get_box_area(box_1)
#         self.box_2_area = self.__get_box_area(box_2)
#
#     @staticmethod
#     def __get_box_min_and_max(box):
#         box_xy = box[..., 0:2]
#         box_wh = box[..., 2:4]
#         box_min = box_xy - box_wh / 2
#         box_max = box_xy + box_wh / 2
#         return box_min, box_max
#
#     @staticmethod
#     def __get_box_area(box):
#         return box[..., 2] * box[..., 3]
#
#     def calculate_iou(self):
#         intersect_min = np.maximum(self.box_1_min, self.box_2_min)
#         intersect_max = np.minimum(self.box_1_max, self.box_2_max)
#         intersect_wh = np.maximum(intersect_max - intersect_min, 0.0)
#         intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
#         union_area = self.box_1_area + self.box_2_area - intersect_area
#         iou = intersect_area / union_area
#         return iou



