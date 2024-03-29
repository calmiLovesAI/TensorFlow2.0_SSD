import tensorflow as tf

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

    @staticmethod
    def _decode(loc, priors, variances):
        boxes = tf.concat(values=[
            priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
            priors[:, 2:] * tf.math.exp(loc[:, 2:] * variances[1])
        ], axis=1)
        min_xy = boxes[:, :2] - boxes[:, 2:] / 2
        max_xy = boxes[:, :2] + boxes[:, 2:] / 2
        return tf.concat(values=[min_xy, max_xy], axis=1)

    # TODO: 怎样选取合适的bounding_boxes
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
            decoded_boxes = InferenceProcedure._decode(loc_data[i], self.priors,
                                                       self.variance)  # (num_priors, 4)  (xmin, ymin, xmax, ymax)格式
            conf_scores = conf_preds[i]  # (num_classes, num_priors)

            t1 = list()
            t1.append(tf.zeros(shape=(self.top_k, 6)))
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
                selected_boxes = tf.gather(params=boxes, indices=selected_indices)
                num_boxes = selected_boxes.shape[0]
                selected_boxes = tf.pad(tensor=selected_boxes, paddings=[[0, self.top_k-num_boxes], [0, 0]])    # (self.top_k, 4)

                selected_scores = tf.expand_dims(tf.gather(params=scores, indices=selected_indices), axis=1)
                selected_scores = tf.pad(tensor=selected_scores, paddings=[[0, self.top_k-num_boxes], [0, 0]])  # (self.top_k, 1)

                selected_classes = tf.fill(dims=[self.top_k, 1], value=cl)
                selected_classes = tf.cast(selected_classes, dtype=tf.float32)

                # (self.top_k, 6(conf, xmin, ymin, xmax, ymax, class_idx))
                targets = tf.concat(values=[selected_scores, selected_boxes, selected_classes],
                                    axis=1)
                t1.append(targets)
            t2 = tf.stack(values=t1, axis=0)
            output.append(t2)
        # (batch_size, C, self.top_k, 6) <dtype: 'float32'>
        output = tf.stack(values=output, axis=0)
        flt = tf.reshape(output, shape=(batch_size, -1, 6))  # (batch_size, self.num_classes * self.top_k, 6)
        idx = tf.argsort(values=flt[:, :, 0], axis=1, direction="DESCENDING")  # (batch_size, self.num_classes * self.top_k,)
        rank = tf.argsort(values=idx, axis=1, direction="ASCENDING")  # (batch_size, self.num_classes * self.top_k,)
        mask = rank < self.top_k
        mask = tf.expand_dims(mask, axis=-1)
        mask = tf.broadcast_to(mask, shape=flt.shape)
        flt = tf.where(condition=mask, x=0, y=flt)
        return tf.reshape(flt, shape=(batch_size, -1, self.top_k, 6))
        # return output

