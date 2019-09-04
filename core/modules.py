import tensorflow as tf


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


def body():
    out = tf.keras.Sequential()
    for num_filter in [16, 32, 64]:
        out.add(down_sample(num_filter))
    return out


def ssd_model(num_anchors, num_classes):
    downsamplers = tf.keras.Sequential()

    for _ in range(3):
        downsamplers.add(down_sample(128))

    class_predictors = tf.keras.Sequential()
    box_predictors = tf.keras.Sequential()

    for _ in range(5):
        class_predictors.add(class_predictor(num_anchors, num_classes))
        box_predictors.add(box_predictor(num_anchors))

    model = tf.keras.Sequential()
    model.add(body(), downsamplers, class_predictors, box_predictors)

    return model
