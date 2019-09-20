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
        y_true = tf.dtypes.cast(y_true, tf.float32)
        y_pred = tf.dtypes.cast(y_pred, tf.float32)
        mask = tf.dtypes.cast(mask, tf.float32)
        return tf.reduce_mean(smooth_l1((y_pred - y_true) * mask))


def smooth_l1(x):
    x = tf.dtypes.cast(x, tf.float32)
    y = tf.where(
        tf.math.greater(1, tf.math.abs(x)),
        0.5 * tf.math.pow(x, 2),
        tf.math.abs(x) - 0.5
    )
    return y


if __name__ == '__main__':
    a = tf.constant([1, 2, 3])
    print(smooth_l1(a))