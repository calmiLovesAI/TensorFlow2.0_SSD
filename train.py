import tensorflow as tf
from core import ssd
from parse_pascal_voc import ParsePascalVOC
from configuration import IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS
from core.label_anchors import LabelAnchors

if __name__ == '__main__':

    # get datasets
    parse = ParsePascalVOC()
    train_dataset, test_dataset, train_count, test_count = parse.split_dataset()

    # initialize model
    model = ssd.SSD()
    model.build(input_shape=(None, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS))
    model.summary()

    # metrics
    class_metric = tf.keras.metrics.Accuracy()
    box_metric = tf.keras.metrics.MeanAbsoluteError()

    # optimizer
    optimizer = tf.keras.optimizers.Adadelta()

    # loss
    train_cls_loss = tf.keras.losses.SparseCategoricalCrossentropy()
    train_reg_loss = tf.keras.losses.MeanSquaredError()
    test_cls_loss = tf.keras.losses.SparseCategoricalCrossentropy()
    test_reg_loss = tf.keras.losses.MeanSquaredError()

    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            anchors, class_preds, box_preds = model(images)
            label_anchors = LabelAnchors(anchors=anchors, labels=labels, class_preds=class_preds)
            a, b, c = label_anchors.get_results()
