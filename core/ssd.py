import tensorflow as tf
from core.modules import class_predictor, box_predictor, concat_predictions, backbone, down_sample_layer
from core import anchor


class SSD(tf.keras.Model):
    def __init__(self, num_classes):
        super(SSD, self).__init__()
        self.sizes = [[0.2, 0.272], [0.37, 0.447], [0.54, 0.619], [0.71, 0.79], [0.88, 0.961]]
        self.ratios = [[1, 2, 0.5]] * 5
        self.num_classes = num_classes
        self.num_anchors = len(self.sizes[0]) + len(self.ratios[0]) - 1

        self.backbone = backbone()

        self.down_sample_1 = down_sample_layer(num_filter=128)
        self.down_sample_2 = down_sample_layer(num_filter=128)
        self.down_sample_3 = down_sample_layer(num_filter=128)

        self.globalmaxpool = tf.keras.layers.GlobalMaxPooling2D()

    def __get_anchors(self, feature_map, sizes, ratios):
        anchor_instantiation = anchor.Anchors(feature_map=feature_map, sizes=sizes, ratios=ratios)
        anchors = anchor_instantiation.get_all_default_boxes()

        return anchors

    def __get_class_preds(self, feature_map):
        class_predictor_block = class_predictor(self.num_anchors, self.num_classes)
        class_preds = class_predictor_block(feature_map).flatten()

        return class_preds

    def __get_box_preds(self, feature_map):
        box_predictor_block = box_predictor(self.num_anchors)
        box_preds = box_predictor_block(feature_map).flatten()

        return box_preds

    def call(self, inputs, training=None, mask=None):
        anchors_list, class_preds_list, box_preds_list = [], [], []
        # stage 1
        x = self.backbone(inputs)
        anchors_list.append(self.__get_anchors(feature_map=x, sizes=self.sizes[0], ratios=self.ratios[0]))
        class_preds_list.append(self.__get_class_preds(feature_map=x))
        box_preds_list.append(self.__get_box_preds(feature_map=x))

        print("Predict scale : ", 0, x.shape, "with", anchors_list[-1].shape[1], "anchors")

        # stage 2
        x = self.down_sample_1(x)
        anchors_list.append(self.__get_anchors(feature_map=x, sizes=self.sizes[1], ratios=self.ratios[1]))
        class_preds_list.append(self.__get_class_preds(feature_map=x))
        box_preds_list.append(self.__get_box_preds(feature_map=x))

        print("Predict scale : ", 1, x.shape, "with", anchors_list[-1].shape[1], "anchors")

        # stage 3
        x = self.down_sample_2(x)
        anchors_list.append(self.__get_anchors(feature_map=x, sizes=self.sizes[2], ratios=self.ratios[2]))
        class_preds_list.append(self.__get_class_preds(feature_map=x))
        box_preds_list.append(self.__get_box_preds(feature_map=x))

        print("Predict scale : ", 2, x.shape, "with", anchors_list[-1].shape[1], "anchors")

        # stage 4
        x = self.down_sample_3(x)
        anchors_list.append(self.__get_anchors(feature_map=x, sizes=self.sizes[3], ratios=self.ratios[3]))
        class_preds_list.append(self.__get_class_preds(feature_map=x))
        box_preds_list.append(self.__get_box_preds(feature_map=x))

        print("Predict scale : ", 3, x.shape, "with", anchors_list[-1].shape[1], "anchors")

        # stage 5
        x = self.globalmaxpool(x)
        anchors_list.append(self.__get_anchors(feature_map=x, sizes=self.sizes[4], ratios=self.ratios[4]))
        class_preds_list.append(self.__get_class_preds(feature_map=x))
        box_preds_list.append(self.__get_box_preds(feature_map=x))

        print("Predict scale : ", 4, x.shape, "with", anchors_list[-1].shape[1], "anchors")

        anchors = concat_predictions(anchors_list)
        class_preds = concat_predictions(class_preds_list)
        box_preds = concat_predictions(box_preds_list)

        class_preds= class_preds.reshape(shape=(0, -1, self.num_classes+1))
        return anchors, class_preds, box_preds
