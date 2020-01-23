import tensorflow as tf
from core.models.resnet import ResNet50


class SSD(tf.keras.Model):
    def __init__(self):
        super(SSD, self).__init__()
        self.backbone = ResNet50()
        self.conv1 = tf.keras.layers.Conv2D(filters=1024, kernel_size=(1, 1), strides=1, padding="same")
        self.conv2_1 = tf.keras.layers.Conv2D(filters=256, kernel_size=(1, 1), strides=1, padding="same")
        self.conv2_2 = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), strides=2, padding="same")
        self.conv3_1 = tf.keras.layers.Conv2D(filters=128, kernel_size=(1, 1), strides=1, padding="same")
        self.conv3_2 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=2, padding="same")
        self.conv4_1 = tf.keras.layers.Conv2D(filters=128, kernel_size=(1, 1), strides=1, padding="same")
        self.conv4_2 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=2, padding="same")
        self.pool = tf.keras.layers.GlobalAveragePooling2D()

    def call(self, inputs, training=None, mask=None):
        branch_1, x = self.backbone(inputs, training=training)

        x = self.conv1(x)
        branch_2 = x

        x = self.conv2_1(x)
        x = self.conv2_2(x)
        branch_3 = x

        x = self.conv3_1(x)
        x = self.conv3_2(x)
        branch_4 = x

        x = self.conv4_1(x)
        x = self.conv4_2(x)
        branch_5 = x

        branch_6 = self.pool(x)

        return branch_1, branch_2, branch_3, branch_4, branch_5, branch_6

