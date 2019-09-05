import tensorflow as tf


class down_sample_layer(tf.keras.layers.Layer):
    def __init__(self, num_filter):
        super(down_sample_layer, self).__init__()
        self.conv = tf.keras.layers.Conv2D(filters=num_filter,
                                           kernel_size=(3, 3),
                                           strides=(1, 1),
                                           padding="same")
        self.bn = tf.keras.layers.BatchNormalization()
        self.activate = tf.keras.layers.Activation(tf.keras.activations.relu)
        self.maxpool = tf.keras.layers.MaxPool2D(pool_size=(2, 2))


    def call(self, inputs, **kwargs):
        x = self.conv(inputs)
        x = self.bn(x)
        x = self.activate(x)
        x = self.maxpool(x)

        return x


def class_predictor(num_anchors, num_classes):
    return tf.keras.layers.Conv2D(filters=num_anchors * (num_classes + 1),
                                  kernel_size=(3, 3),
                                  strides=(1, 1),
                                  padding="same")


def box_predictor(num_anchors):
    return tf.keras.layers.Conv2D(filters=num_anchors * 4,
                                  kernel_size=(3, 3),
                                  strides=(1, 1),
                                  padding="same")


def down_sample(num_filters):
    x = tf.keras.Sequential()
    for _ in range(2):
        x.add(tf.keras.layers.Conv2D(filters=num_filters,
                                     kernel_size=(3, 3),
                                     strides=(1, 1),
                                     padding="same"))
        x.add(tf.keras.layers.BatchNormalization())
        x.add(tf.keras.layers.Activation(tf.keras.activations.relu))
    x.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))

    return x


def concat_predictions(preds):
    return tf.concat(values=preds, axis=1)


def backbone():
    out = tf.keras.Sequential()
    for num_filter in [16, 32, 64]:
        out.add(down_sample(num_filter))
    return out


