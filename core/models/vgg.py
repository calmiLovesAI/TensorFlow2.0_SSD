import tensorflow as tf


class VGG(tf.keras.layers.Layer):
    def __init__(self, use_bn=False):
        super(VGG, self).__init__()
        self.conv1 = VGG._make_conv_block(64, 3, 1, "same", use_bn)
        self.conv2 = VGG._make_conv_block(64, 3, 1, "same", use_bn)
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding="same")

        self.conv3 = VGG._make_conv_block(128, 3, 1, "same", use_bn)
        self.conv4 = VGG._make_conv_block(128, 3, 1, "same", use_bn)
        self.pool2 = tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding="same")

        self.conv5 = VGG._make_conv_block(256, 3, 1, "same", use_bn)
        self.conv6 = VGG._make_conv_block(256, 3, 1, "same", use_bn)
        self.conv7 = VGG._make_conv_block(256, 3, 1, "same", use_bn)
        self.pool3 = tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding="same")

        self.conv8 = VGG._make_conv_block(512, 3, 1, "same", use_bn)
        self.conv9 = VGG._make_conv_block(512, 3, 1, "same", use_bn)
        self.conv10 = VGG._make_conv_block(512, 3, 1, "same", use_bn)
        self.pool4 = tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding="same")

        self.conv11 = VGG._make_conv_block(512, 3, 1, "same", use_bn)
        self.conv12 = VGG._make_conv_block(512, 3, 1, "same", use_bn)
        self.conv13 = VGG._make_conv_block(512, 3, 1, "same", use_bn)

        self.pool5 = tf.keras.layers.MaxPool2D(pool_size=3, strides=1, padding="same")
        self.conv14 = tf.keras.layers.Conv2D(filters=1024, kernel_size=3, strides=1, padding="same", dilation_rate=6)
        self.conv15 = tf.keras.layers.Conv2D(filters=1024, kernel_size=1, strides=1, padding="same")

    @staticmethod
    def _make_conv_block(out_channels, kernel_size, strides, padding, use_bn=True):
        if use_bn:
            return tf.keras.Sequential([
                tf.keras.layers.Conv2D(filters=out_channels, kernel_size=kernel_size, strides=strides, padding=padding),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU()
            ])
        else:
            return tf.keras.Sequential([
                tf.keras.layers.Conv2D(filters=out_channels, kernel_size=kernel_size, strides=strides, padding=padding),
                tf.keras.layers.ReLU()
            ])

    def call(self, inputs, training=None, *args, **kwargs):
        x = self.conv1(inputs, training=training)
        x = self.conv2(x, training=training)
        x = self.pool1(x)

        x = self.conv3(x, training=training)
        x = self.conv4(x, training=training)
        x = self.pool2(x)

        x = self.conv5(x, training=training)
        x = self.conv6(x, training=training)
        x = self.conv7(x, training=training)
        x = self.pool3(x)

        x = self.conv8(x, training=training)
        x = self.conv9(x, training=training)
        x = self.conv10(x, training=training)
        o1 = x
        x = self.pool4(x)

        x = self.conv11(x, training=training)
        x = self.conv12(x, training=training)
        x = self.conv13(x, training=training)

        x = self.pool5(x)
        x = self.conv14(x)
        x = tf.nn.relu(x)
        x = self.conv15(x)
        x = tf.nn.relu(x)
        o2 = x

        return o1, o2
