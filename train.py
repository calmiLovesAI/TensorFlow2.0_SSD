import tensorflow as tf
from core import ssd


if __name__ == '__main__':

    # get datasets


    # initialize model
    model = ssd.SSD()

    class_metric = tf.keras.metrics.Accuracy()
    box_metric = tf.keras.metrics.MeanAbsoluteError()
