import tensorflow as tf
from configuration import alpha, gamma


class FocalLoss(tf.keras.losses.Loss):
    def __init__(self):
        super(FocalLoss, self).__init__()
        pass

    def call(self, y_true, y_pred):
        pass


class SmoothL1Loss(tf.keras.losses.Loss):
    def __init__(self):
        super(SmoothL1Loss, self).__init__()

    def __call__(self, y_true, y_pred, mask):
        return tf.reduce_mean(smooth_l1((y_pred - y_true) * mask))


def smooth_l1(x):
    if tf.math.abs(x) < 1:
        y = 0.5 * tf.math.pow(x, 2)
    else:
        y = tf.math.abs(x) - 0.5
    return y
