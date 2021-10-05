import tensorflow as tf
import numpy as np
from configuration import VARIANCE, NMS_THRESHOLD, CONFIDENCE_THRESHOLD, MAX_BOXES_NUM
from core.anchor import DefaultBoxes


class InferenceProcedure:
    def __init__(self, model, num_classes):
        self.model = model
        self.priors = DefaultBoxes().generate_boxes()  # (num_priors,4)  num_priors表示anchor总数，在SSD300中为8732
        self.top_k = MAX_BOXES_NUM
        self.num_classes = num_classes
        self.variance = VARIANCE
        self.conf_thresh = CONFIDENCE_THRESHOLD
        self.nms_thresh = NMS_THRESHOLD

    def _decode(self, loc, priors, variances):
        boxes = tf.concat(values=[
            priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
            priors[:, 2:] * tf.math.exp(loc[:, 2:] * variances[1])
        ], axis=1)
        min_xy = boxes[:, :2] - boxes[:, 2:] / 2
        max_xy = boxes[:, 2:] + boxes[:, :2]
        return tf.concat(values=[min_xy, max_xy], axis=1)

    def __call__(self, inputs, *args, **kwargs):
        # loc_data: (batch_size, num_priors, 4)
        # conf_data: (batch_size, num_priors, num_classes)
        loc_data, conf_data = self.model(inputs, training=False)
        conf_data = tf.nn.softmax(conf_data)
        batch_size = loc_data.shape[0]
        num_priors = self.priors.shape[0]
        # output = tf.zeros(shape=(batch_size, self.num_classes, self.top_k, 5))
        conf_preds = tf.transpose(a=conf_data, perm=[0, 2, 1])  # (batch_size, num_classes, num_priors)

        # 解码
        output = list()
        for i in range(batch_size):
            decoded_boxes = self._decode(loc_data[i], self.priors, self.variance)  # (num_priors, 4)  (xmin, ymin, xmax, ymax)格式
            conf_scores = conf_preds[i]   # (num_classes, num_priors)

            t1 = list()
            t1.append(tf.zeros(shape=(self.top_k, 5)))
            for cl in range(1, self.num_classes):
                # shape: (num_priors,)  dtype: bool
                c_mask = tf.math.greater(conf_scores[cl], self.conf_thresh)
                scores = tf.boolean_mask(conf_scores[cl], c_mask)
                if scores.shape[0] == 0:
                    continue
                # shape: (num_priors, 1)  dtype: bool
                l_mask = tf.broadcast_to(tf.expand_dims(c_mask, axis=1), shape=decoded_boxes.shape)
                # shape: (num_boxes, 4)
                boxes = tf.reshape(tf.boolean_mask(decoded_boxes, l_mask), shape=(-1, 4))
                selected_indices = tf.image.non_max_suppression(boxes=boxes,
                                                                scores=scores,
                                                                max_output_size=self.top_k,
                                                                iou_threshold=self.nms_thresh)
                selected_boxes = tf.gather(params=boxes, indices=selected_indices)  # (self.top_k, 4)
                selected_scores = tf.gather(params=scores, indices=selected_indices)  # (self.top_k,)
                # (self.top_k, 5(conf, xmin, ymin, xmax, ymax))
                targets = tf.concat(values=[tf.expand_dims(selected_scores, axis=1), selected_boxes], axis=1)
                t1.append(targets)
            t1 = tf.stack(values=t1, axis=0)
            output.append(t1)
        # (batch_size, self.num_classes, self.top_k, 5) <dtype: 'float32'>
        output = tf.stack(values=output, axis=0)
        flt = tf.reshape(output, shape=(batch_size, -1, 5))  # (batch_size, self.num_classes * self.top_k, 5)
        idx = tf.argsort(values=flt[:, :, 0], axis=1, direction="DESCENDING") # (batch_size, self.num_classes * self.top_k,)
        rank = tf.argsort(values=idx, axis=1, direction="ASCENDING")  # (batch_size, self.num_classes * self.top_k,)
        mask = rank < self.top_k
        mask = tf.expand_dims(mask, axis=-1)
        mask = tf.broadcast_to(mask, shape=flt.shape)
        flt = tf.where(condition=mask, x=0, y=flt)
        return flt





# class InferenceProcedure(object):
#     def __init__(self, model):
#         self.model = model
#         self.num_classes = NUM_CLASSES
#         self.image_size = np.array([IMAGE_HEIGHT, IMAGE_WIDTH], dtype=np.float32)
#         self.nms_op = NMS()
#
#     def __get_ssd_prediction(self, image):
#         output = self.model(image, training=False)
#         pred = ssd_prediction(feature_maps=output, num_classes=self.num_classes)
#         return pred, output
#
#     @staticmethod
#     def __resize_boxes(boxes, image_height, image_width):
#         cx = boxes[..., 0] * image_width
#         cy = boxes[..., 1] * image_height
#         w = boxes[..., 2] * image_width
#         h = boxes[..., 3] * image_height
#         xmin = cx - w / 2
#         ymin = cy - h / 2
#         xmax = cx + w / 2
#         ymax = cy + h / 2
#         resized_boxes = tf.stack(values=[xmin, ymin, xmax, ymax], axis=-1)
#         return resized_boxes
#
#     def __filter_background_boxes(self, ssd_predict_boxes):
#         is_object_exist = True
#         num_of_total_predict_boxes = ssd_predict_boxes.shape[1]
#         # scores = tf.nn.softmax(ssd_predict_boxes[..., :self.num_classes])
#         scores = ssd_predict_boxes[..., :self.num_classes]
#         classes = tf.math.argmax(input=scores, axis=-1)
#         filtered_boxes_list = []
#         for i in range(num_of_total_predict_boxes):
#             if classes[:, i] != 0:
#                 filtered_boxes_list.append(ssd_predict_boxes[:, i, :])
#         if filtered_boxes_list:
#             filtered_boxes = tf.stack(values=filtered_boxes_list, axis=1)
#             return is_object_exist, filtered_boxes
#         else:
#             is_object_exist = False
#             return is_object_exist, ssd_predict_boxes
#
#     def __offsets_to_true_coordinates(self, pred_boxes, ssd_output):
#         pred_classes = tf.reshape(tensor=pred_boxes[..., :self.num_classes], shape=(-1, self.num_classes))
#         pred_coords = tf.reshape(tensor=pred_boxes[..., self.num_classes:], shape=(-1, 4))
#         default_boxes = DefaultBoxes(feature_map_list=ssd_output).generate_default_boxes()
#         d_cx, d_cy, d_w, d_h = default_boxes[:, 0:1], default_boxes[:, 1:2], default_boxes[:, 2:3], default_boxes[:, 3:4]
#         offset_cx, offset_cy, offset_w, offset_h = pred_coords[:, 0:1], pred_coords[:, 1:2], pred_coords[:, 2:3], pred_coords[:, 3:4]
#         true_cx = offset_cx * d_w + d_cx
#         true_cy = offset_cy * d_h + d_cy
#         true_w = tf.math.exp(offset_w) * d_w
#         true_h = tf.math.exp(offset_h) * d_h
#         true_coords = tf.concat(values=[true_cx, true_cy, true_w, true_h], axis=-1)
#         true_classes_and_coords = tf.concat(values=[pred_classes, true_coords], axis=-1)
#         true_classes_and_coords = tf.expand_dims(input=true_classes_and_coords, axis=0)
#         return true_classes_and_coords
#
#     def get_final_boxes(self, image):
#         pred_boxes, ssd_output = self.__get_ssd_prediction(image)
#         pred_boxes = self.__offsets_to_true_coordinates(pred_boxes=pred_boxes, ssd_output=ssd_output)
#         is_object_exist, filtered_pred_boxes = self.__filter_background_boxes(pred_boxes)
#         if is_object_exist:
#             # scores = tf.nn.softmax(filtered_pred_boxes[..., :self.num_classes])
#             scores = filtered_pred_boxes[..., :self.num_classes]
#             pred_boxes_scores = tf.reshape(tensor=scores, shape=(-1, self.num_classes))
#             pred_boxes_coord = filtered_pred_boxes[..., self.num_classes:]
#             pred_boxes_coord = tf.reshape(tensor=pred_boxes_coord, shape=(-1, 4))
#             resized_pred_boxes = self.__resize_boxes(boxes=pred_boxes_coord,
#                                                      image_height=image.shape[1],
#                                                      image_width=image.shape[2])
#             box_tensor, score_tensor, class_tensor = self.nms_op.nms(boxes=resized_pred_boxes,
#                                                                      box_scores=pred_boxes_scores)
#             return is_object_exist, box_tensor, score_tensor, class_tensor
#         else:
#             return is_object_exist, tf.zeros(shape=(1, 4)), tf.zeros(shape=(1,)), tf.zeros(shape=(1,))