import tensorflow as tf

from core.anchor import DefaultBoxes
from utils.IoU import match
from configuration import VARIANCE
from utils.tf_functions import log_sum_exp, clip_by_value
from utils.tools import true_coords_labels


class MultiBoxLoss():
    def __init__(self, num_classes, overlap_thresh, neg_pos):
        self.default_boxes = tf.convert_to_tensor(DefaultBoxes().generate_boxes())  # Tensor, shape: (先验框总数(8732), 4)
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.variance = VARIANCE
        self.negpos_ratio = neg_pos

    def __call__(self, y_true, y_pred):
        """

        :param y_true:  Tensor, shape: (batch_size, MAX_BOXES_PER_IMAGE, 5(xmin, ymin, xmax, ymax, class_index))
        :param y_pred:  Tuple, (loc, conf), 其中loc的shape是(batch_size, 先验框总数, 4), conf的shape是(batch_size, 先验框总数, 21)
        :return:
        """
        loc_data, conf_data = y_pred
        num = loc_data.shape[0]
        priors = self.default_boxes[:loc_data.shape[1], :]
        num_priors = (priors.shape[0])
        num_classes = self.num_classes

        # 匹配先验框和GT boxes
        loc_t = []
        conf_t = []
        for idx in range(num):
            # truths = y_true[idx][:, :-1]
            # labels = y_true[idx][:, -1]
            truths, labels = true_coords_labels(idx, y_true)
            match(threshold=self.threshold, truths=truths, priors=priors,
                  variances=self.variance, labels=labels, loc_t=loc_t, conf_t=conf_t)
        loc_t = tf.stack(values=loc_t, axis=0)
        conf_t = tf.stack(values=conf_t, axis=0)

        pos = conf_t > 0
        # num_pos = tf.math.reduce_sum(input_tensor=tf.cast(pos, dtype=tf.int32), axis=1, keepdims=True)

        # 位置loss: Smooth L1 loss
        pos_idx = tf.expand_dims(pos, axis=-1)
        pos_idx = tf.broadcast_to(pos_idx, shape=loc_data.shape)
        loc_p = tf.boolean_mask(tensor=loc_data, mask=pos_idx)
        loc_t = tf.boolean_mask(tensor=loc_t, mask=pos_idx)
        smooth_l1_loss_fn = tf.keras.losses.Huber(delta=1.0, reduction=tf.keras.losses.Reduction.SUM)
        loss_l = smooth_l1_loss_fn(y_true=loc_t, y_pred=loc_p)

        batch_conf = tf.reshape(conf_data, shape=(-1, self.num_classes))
        conf_t = tf.cast(conf_t, dtype=tf.int32)
        loss_c = log_sum_exp(batch_conf) - tf.gather(params=batch_conf, indices=tf.reshape(conf_t, shape=(-1, 1)),
                                                     batch_dims=1)

        # Hard Negative Mining
        loss_c = tf.reshape(loss_c, shape=(num, -1))
        loss_c = tf.where(condition=pos, x=0, y=loss_c)
        loss_c = tf.reshape(loss_c, shape=(num, -1))
        loss_idx = tf.argsort(values=loss_c, axis=1, direction="DESCENDING")
        idx_rank = tf.argsort(values=loss_idx, axis=1, direction="ASCENDING")
        num_pos = tf.reduce_sum(tf.cast(pos, dtype=tf.int32), axis=1, keepdims=True)
        num_neg = clip_by_value(t=self.negpos_ratio * num_pos, clip_value_max=pos.shape[1] - 1)
        neg = idx_rank < num_neg

        # 置信度loss
        pos_idx = tf.broadcast_to(tf.expand_dims(pos, axis=2), shape=conf_data.shape)
        neg_idx = tf.broadcast_to(tf.expand_dims(neg, axis=2), shape=conf_data.shape)
        conf_p = tf.boolean_mask(tensor=conf_data, mask=tf.math.logical_or(pos_idx, neg_idx))
        conf_p = tf.reshape(conf_p, shape=(-1, self.num_classes))
        targets_weighted = tf.boolean_mask(tensor=conf_t, mask=tf.math.logical_or(pos, neg))
        ce_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.SUM)
        loss_c = ce_fn(y_true=targets_weighted, y_pred=conf_p)

        N = tf.reduce_sum(num_pos)
        N = tf.cast(N, dtype=tf.float32)
        loss_l /= N
        loss_c /= N
        return loss_l, loss_c

# from configuration import reg_loss_weight, NUM_CLASSES, alpha, gamma
# from utils.focal_loss import sigmoid_focal_loss
#
#
# class SmoothL1Loss:
#     def __init__(self):
#         pass
#
#     def __call__(self, y_true, y_pred, *args, **kwargs):
#         absolute_value = tf.math.abs(y_true - y_pred)
#         mask_boolean = tf.math.greater_equal(x=absolute_value, y=1.0)
#         mask_float32 = tf.cast(x=mask_boolean, dtype=tf.float32)
#         smooth_l1_loss = (1.0 - mask_float32) * 0.5 * tf.math.square(absolute_value) + mask_float32 * (absolute_value - 0.5)
#         return smooth_l1_loss
#
#
# class SSDLoss:
#     def __init__(self):
#         self.smooth_l1_loss = SmoothL1Loss()
#         self.reg_loss_weight = reg_loss_weight
#         self.cls_loss_weight = 1 - reg_loss_weight
#         self.num_classes = NUM_CLASSES
#
#     @staticmethod
#     def __cover_background_boxes(true_boxes):
#         symbol = true_boxes[..., -1]
#         mask_symbol = tf.where(symbol < 0.5, 0.0, 1.0)
#         mask_symbol = tf.expand_dims(input=mask_symbol, axis=-1)
#         cover_boxes_tensor = tf.tile(input=mask_symbol, multiples=tf.constant([1, 1, 4], dtype=tf.dtypes.int32))
#         return cover_boxes_tensor
#
#     def __call__(self, y_true, y_pred, *args, **kwargs):
#         # y_true : tensor, shape: (batch_size, total_num_of_default_boxes, 5)
#         # y_pred : tensor, shape: (batch_size, total_num_of_default_boxes, NUM_CLASSES + 4)
#         true_class = tf.cast(x=y_true[..., -1], dtype=tf.dtypes.int32)
#         pred_class = y_pred[..., :self.num_classes]
#         true_class_onehot = tf.one_hot(indices=true_class, depth=self.num_classes, axis=-1)
#         class_loss_value = tf.math.reduce_sum(sigmoid_focal_loss(y_true=true_class_onehot, y_pred=pred_class, alpha=alpha, gamma=gamma))
#
#         cover_boxes = self.__cover_background_boxes(true_boxes=y_true)
#         # true_coord = y_true[..., :4] * cover_boxes
#         # pred_coord = y_pred[..., self.num_classes:] * cover_boxes
#         true_coord = y_true[..., :4]
#         pred_coord = y_pred[..., self.num_classes:]
#         reg_loss_value = tf.math.reduce_sum(self.smooth_l1_loss(y_true=true_coord, y_pred=pred_coord) * cover_boxes)
#
#         loss = self.cls_loss_weight * class_loss_value + self.reg_loss_weight * reg_loss_value
#         return loss, class_loss_value, reg_loss_value
