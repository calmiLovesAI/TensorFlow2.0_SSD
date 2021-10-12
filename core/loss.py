import tensorflow as tf

from core.anchor import DefaultBoxes
from utils.IoU import match
from configuration import VARIANCE
from utils.tf_functions import log_sum_exp, clip_by_value
from utils.tools import true_coords_labels


class MultiBoxLoss:
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
        loc_p = tf.reshape(loc_p, shape=(-1, 4))
        loc_t = tf.reshape(loc_t, shape=(-1, 4))
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

