import tensorflow as tf

from core.models.vgg import VGG

from configuration import NUM_CLASSES, STAGE_BOXES_PER_PIXEL


class L2Normalize(tf.keras.layers.Layer):
    def __init__(self, epsilon):
        super(L2Normalize, self).__init__()
        self.epsilon = epsilon

    def build(self, input_shape):
        self.w = self.add_weight(name="w",
                                 shape=(1, 1, 1, input_shape[-1]),
                                 initializer=tf.keras.initializers.RandomNormal(),
                                 dtype=tf.float32,
                                 trainable=True)

    def call(self, inputs, *args, **kwargs):
        return self.w * tf.math.l2_normalize(x=inputs, axis=-1, epsilon=self.epsilon)


class ExtraLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(ExtraLayer, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=256, kernel_size=1, strides=1, padding="same")
        self.conv2 = tf.keras.layers.Conv2D(filters=512, kernel_size=3, strides=2, padding="same")
        self.conv3 = tf.keras.layers.Conv2D(filters=128, kernel_size=1, strides=1, padding="same")
        self.conv4 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=2, padding="same")
        self.conv5 = tf.keras.layers.Conv2D(filters=128, kernel_size=1, strides=1, padding="same")
        self.conv6 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=1, padding="valid")
        self.conv7 = tf.keras.layers.Conv2D(filters=128, kernel_size=1, strides=1, padding="same")
        self.conv8 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=1, padding="valid")

    def call(self, inputs, *args, **kwargs):
        x = self.conv1(inputs)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = tf.nn.relu(x)
        o1 = x
        x = self.conv3(x)
        x = tf.nn.relu(x)
        x = self.conv4(x)
        x = tf.nn.relu(x)
        o2 = x
        x = self.conv5(x)
        x = tf.nn.relu(x)
        x = self.conv6(x)
        x = tf.nn.relu(x)
        o3 = x
        x = self.conv7(x)
        x = tf.nn.relu(x)
        x = self.conv8(x)
        x = tf.nn.relu(x)
        o4 = x
        return o1, o2, o3, o4


class SSD(tf.keras.Model):
    def __init__(self):
        super(SSD, self).__init__()
        self.num_classes = NUM_CLASSES
        self.stage_boxes_per_pixel = STAGE_BOXES_PER_PIXEL

        self.backbone = VGG(use_bn=True)
        self.l2_norm = L2Normalize(epsilon=1e-10)
        self.extras = ExtraLayer()
        self.locs, self.confs = self._make_loc_conf(num_classes=self.num_classes)

    def _make_loc_conf(self, num_classes):
        loc_layers = list()
        conf_layers = list()
        for i in self.stage_boxes_per_pixel:
            loc_layers.append(tf.keras.layers.Conv2D(filters=i * 4, kernel_size=3, strides=1, padding="same"))
            conf_layers.append(
                tf.keras.layers.Conv2D(filters=i * num_classes, kernel_size=3, strides=1, padding="same"))

        return loc_layers, conf_layers

    def call(self, inputs, training=None, mask=None):
        sources = list()
        loc = list()
        conf = list()

        x1, x = self.backbone(inputs, training=training)
        x1 = self.l2_norm(x1)
        sources.append(x1)
        sources.append(x)

        o1, o2, o3, o4 = self.extras(x)
        sources.extend([o1, o2, o3, o4])
        x = o4

        for (x, l, c) in zip(sources, self.locs, self.confs):
            loc.append(l(x))
            conf.append(c(x))

        loc = tf.concat(values=[tf.reshape(o, shape=(o.shape[0], -1)) for o in loc], axis=1)
        conf = tf.concat(values=[tf.reshape(o, shape=(o.shape[0], -1)) for o in conf], axis=1)

        loc = tf.reshape(loc, shape=(loc.shape[0], -1, 4))
        conf = tf.reshape(conf, shape=(conf.shape[0], -1, self.num_classes))

        return loc, conf
