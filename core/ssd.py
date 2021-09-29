import tensorflow as tf
from core.models.resnet import ResNet50
from core.models.vgg import VGG

from configuration import NUM_CLASSES, ASPECT_RATIOS


class L2Normalize(tf.keras.layers.Layer):
    def __init__(self, epsilon):
        super(L2Normalize, self).__init__()
        self.epsilon = epsilon

    def build(self, input_shape):
        self.w = self.add_weight(shape=(1, 1, 1, input_shape[-1]),
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
        self.conv6 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=1, padding="same")
        self.conv7 = tf.keras.layers.Conv2D(filters=128, kernel_size=1, strides=1, padding="same")
        self.conv8 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=1, padding="same")

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
        self.backbone = VGG(use_bn=True)
        self.l2_norm = L2Normalize(epsilon=1e-10)
        self.extras = ExtraLayer()
        self.locs, self.confs = self._make_loc_conf(num_classes=self.num_classes)

    def _make_loc_conf(self, num_classes):
        loc_layers = list()
        conf_layers = list()
        params = [4, 6, 6, 6, 4, 4]
        for i in params:
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

# class SSD(tf.keras.Model):
#     def __init__(self):
#         super(SSD, self).__init__()
#         self.num_classes = NUM_CLASSES
#         self.anchor_ratios = ASPECT_RATIOS
#
#         self.backbone = ResNet50()
#         self.learnable_factor = self.add_weight(shape=(1, 1, 1, 512), dtype=tf.float32, initializer=tf.keras.initializers.Ones(), trainable=True)
#         # self.conv1 = tf.keras.layers.Conv2D(filters=1024, kernel_size=(1, 1), strides=1, padding="same")
#         self.conv2_1 = tf.keras.layers.Conv2D(filters=256, kernel_size=(1, 1), strides=1, padding="same")
#         self.conv2_2 = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), strides=2, padding="same")
#         self.conv3_1 = tf.keras.layers.Conv2D(filters=128, kernel_size=(1, 1), strides=1, padding="same")
#         self.conv3_2 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=2, padding="same")
#         self.conv4_1 = tf.keras.layers.Conv2D(filters=128, kernel_size=(1, 1), strides=1, padding="same")
#         self.conv4_2 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=2, padding="same")
#         self.pool = tf.keras.layers.GlobalAveragePooling2D()
#
#         self.predict_1 = self._predict_layer(k=self._get_k(i=0))
#         self.predict_2 = self._predict_layer(k=self._get_k(i=1))
#         self.predict_3 = self._predict_layer(k=self._get_k(i=2))
#         self.predict_4 = self._predict_layer(k=self._get_k(i=3))
#         self.predict_5 = self._predict_layer(k=self._get_k(i=4))
#         self.predict_6 = self._predict_layer(k=self._get_k(i=5))
#
#     def _predict_layer(self, k):
#         filter_num = k * (self.num_classes + 4)
#         return tf.keras.layers.Conv2D(filters=filter_num, kernel_size=(3, 3), strides=1, padding="same")
#
#     def _get_k(self, i):
#         # k is the number of boxes generated at each position of the feature map.
#         return len(self.anchor_ratios[i]) + 1
#
#     def call(self, inputs, training=None, mask=None):
#         branch_1, x = self.backbone(inputs, training=training)
#         branch_1 = tf.math.l2_normalize(x=branch_1, axis=-1, epsilon=1e-12) * self.learnable_factor
#         predict_1 = self.predict_1(branch_1)
#
#         # x = self.conv1(x)
#         branch_2 = x
#         predict_2 = self.predict_2(branch_2)
#
#         x = tf.nn.relu(self.conv2_1(x))
#         x = tf.nn.relu(self.conv2_2(x))
#         branch_3 = x
#         predict_3 = self.predict_3(branch_3)
#
#         x = tf.nn.relu(self.conv3_1(x))
#         x = tf.nn.relu(self.conv3_2(x))
#         branch_4 = x
#         predict_4 = self.predict_4(branch_4)
#
#         x = tf.nn.relu(self.conv4_1(x))
#         x = tf.nn.relu(self.conv4_2(x))
#         branch_5 = x
#         predict_5 = self.predict_5(branch_5)
#
#         branch_6 = self.pool(x)
#         branch_6 = tf.expand_dims(input=branch_6, axis=1)
#         branch_6 = tf.expand_dims(input=branch_6, axis=2)
#         predict_6 = self.predict_6(branch_6)
#
#         # predict_i shape : (batch_size, h, w, k * (c+4)), where c is self.num_classes.
#         # h == w == [38, 19, 10, 5, 3, 1] for predict_i (i: 1~6)
#         return [predict_1, predict_2, predict_3, predict_4, predict_5, predict_6]
#
#
# def ssd_prediction(feature_maps, num_classes):
#     batch_size = feature_maps[0].shape[0]
#     predicted_features_list = []
#     for feature in feature_maps:
#         predicted_features_list.append(tf.reshape(tensor=feature, shape=(batch_size, -1, num_classes + 4)))
#     predicted_features = tf.concat(values=predicted_features_list, axis=1)  # shape: (batch_size, 8732, (c+4))
#     return predicted_features
