import tensorflow as tf


class SmoothL1Loss(object):
    def __init__(self):
        pass

    def __call__(self, y_true, y_pred, *args, **kwargs):
        absolute_value = tf.math.abs(y_true - y_pred)
        mask_boolean = tf.math.greater_equal(x=absolute_value, y=1.0)
        mask_float32 = tf.cast(x=mask_boolean, dtype=tf.float32)
        smooth_l1_loss = (1.0 - mask_float32) * 0.5 * tf.math.square(absolute_value) + mask_float32 * (absolute_value - 0.5)
        return tf.math.reduce_mean(tf.math.reduce_sum(smooth_l1_loss))


class SSDLoss(object):
    def __init__(self):
        pass

    def __call__(self, y_true, y_pred, *args, **kwargs):
        print(y_true.shape, y_pred.shape)