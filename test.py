import tensorflow as tf
import cv2
import os

from configuration import OBJECT_CLASSES, save_model_dir, test_picture_dir, NUM_CLASSES
from core.inference import InferenceProcedure
from core.ssd import SSD
from utils.tools import preprocess_image, resize_box


def find_class_name(class_id):
    id2class = dict((v, k) for k, v in OBJECT_CLASSES.items())
    return id2class[class_id]


# shape of boxes : (N, 4)  (xmin, ymin, xmax, ymax)
# shape of scores : (N,)
# shape of classes : (N,)
def draw_boxes_on_image(image, boxes, scores, classes):
    num_boxes = boxes.shape[0]
    for i in range(num_boxes):
        class_and_score = str(find_class_name(classes[i])) + ": " + str(scores[i])
        cv2.rectangle(img=image, pt1=(boxes[i, 0], boxes[i, 1]), pt2=(boxes[i, 2], boxes[i, 3]), color=(255, 0, 0),
                      thickness=2)
        cv2.putText(img=image, text=class_and_score, org=(boxes[i, 0], boxes[i, 1] - 10),
                    fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1.5, color=(0, 255, 255), thickness=2)
    return image


def test_single_picture(picture_dir, model):
    image_tensor = preprocess_image(picture_dir)
    image_tensor = tf.expand_dims(image_tensor, axis=0)
    image_array = cv2.imread(picture_dir)
    h, w, _ = image_array.shape
    procedure = InferenceProcedure(model=model, num_classes=NUM_CLASSES)
    results = procedure(image_tensor)
    results = tf.squeeze(results, axis=0)
    # 筛选置信度高于某一值的框
    filter_mask = results[:, :, 0] > 0.6
    filter_mask = tf.expand_dims(filter_mask, axis=-1)
    filter_mask = tf.broadcast_to(filter_mask, shape=results.shape)
    results = tf.boolean_mask(results, filter_mask)
    results = tf.reshape(results, shape=(-1, 6))

    scores = results[:, 0].numpy()
    boxes = results[:, 1: 5].numpy()
    boxes = resize_box(boxes, h, w)
    classes = tf.cast(results[:, -1] - 1, dtype=tf.int32).numpy()

    image_with_boxes = draw_boxes_on_image(image_array, boxes, scores, classes)

    return image_with_boxes


if __name__ == '__main__':
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)

    ssd_model = SSD()
    # 始终加载最新的权重文件
    last_epoch = os.listdir(save_model_dir)[-2].split(".")[0]
    ssd_model.load_weights(filepath=save_model_dir + last_epoch)

    image = test_single_picture(picture_dir=test_picture_dir, model=ssd_model)

    cv2.namedWindow("detect result", flags=cv2.WINDOW_NORMAL)
    cv2.imshow("detect result", image)
    cv2.waitKey(0)
