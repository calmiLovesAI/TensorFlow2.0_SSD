import tensorflow as tf
import numpy as np
from configuration import reg_loss_weight


class SmoothL1Loss(object):
    def __init__(self):
        pass

    def __call__(self, y_true, y_pred, *args, **kwargs):
        absolute_value = tf.math.abs(y_true - y_pred)
        mask_boolean = tf.math.greater_equal(x=absolute_value, y=1.0)
        mask_float32 = tf.cast(x=mask_boolean, dtype=tf.float32)
        smooth_l1_loss = (1.0 - mask_float32) * 0.5 * tf.math.square(absolute_value) + mask_float32 * (absolute_value - 0.5)
        return tf.math.reduce_sum(smooth_l1_loss)


class SSDLoss(object):
    def __init__(self):
        self.smooth_l1_loss = SmoothL1Loss()
        self.reg_loss_weight = reg_loss_weight
        self.cls_loss_weight = 1 - reg_loss_weight

    @staticmethod
    def __cover_background_boxes(true_boxes):
        batch_size, total_num_of_boxes, _ = true_boxes.shape
        cover_boxes = np.ones_like(true_boxes, dtype=np.float32)
        for b in range(batch_size):
            for t in range(total_num_of_boxes):
                if true_boxes[b, t, -1] == 0.0:
                    cover_boxes[b, t, :] = np.zeros_like(true_boxes[b, t, :], dtype=np.float32)
        cover_boxes_tensor = tf.convert_to_tensor(value=cover_boxes[..., :4], dtype=tf.dtypes.float32)
        return cover_boxes_tensor

    def __call__(self, y_true, y_pred, *args, **kwargs):
        # y_true : tensor, shape: (batch_size, total_num_of_default_boxes, 5)
        # y_pred : tensor, shape: (batch_size, total_num_of_default_boxes, 25)
        true_class = tf.cast(x=y_true[..., -1], dtype=tf.dtypes.int32)
        pred_class = y_pred[..., :21]
        class_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=true_class, logits=pred_class)
        class_loss_value = tf.math.reduce_mean(class_loss)

        cover_boxes = self.__cover_background_boxes(true_boxes=y_true)
        true_coord = y_true[..., :4] * cover_boxes
        pred_coord = y_pred[..., 21:] * cover_boxes
        reg_loss_value = self.smooth_l1_loss(y_true=true_coord, y_pred=pred_coord)

        loss = self.cls_loss_weight * class_loss_value + self.reg_loss_weight * reg_loss_value
        return loss
