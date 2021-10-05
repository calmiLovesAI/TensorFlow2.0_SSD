import tensorflow as tf
import numpy as np

from configuration import CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH


# def x_y_meshgrid(x_row, y_col):
#     x = np.arange(0, x_row)
#     y = np.arange(0, y_col)
#     X, Y = np.meshgrid(x, y)
#     X = X.flatten()
#     Y = Y.flatten()
#     return X, Y


def true_coords_labels(idx, y_true):
    """
    去除标签中的padding部分，只保留真实部分
    :param idx:
    :param y_true:
    :return:
    """
    y_true = y_true[idx]
    mask = y_true[:, -1] >= 0
    y_true = tf.boolean_mask(y_true, mask)
    true_coords = y_true[:, :-1]
    true_labels = y_true[:, -1]
    return true_coords, true_labels


def preprocess_image(img_path):
    # read pictures
    img_raw = tf.io.read_file(img_path)
    # decode pictures
    img_tensor = tf.io.decode_image(contents=img_raw, channels=CHANNELS, dtype=tf.dtypes.float32)
    # resize
    img_tensor = tf.image.resize(img_tensor, [IMAGE_HEIGHT, IMAGE_WIDTH])
    return img_tensor


def image_box_transform(image, boxes):
    """

    :param image: str, 图片路径
    :param boxes: numpy.ndarray, shape: (MAX_BOXES_PER_IMAGE, 5)
    :return:
    """
    image_tensor = preprocess_image(image)
    h, w, _ = image_tensor.shape
    # boxes坐标归一化到0~1范围内
    boxes[:, 0] /= w
    boxes[:, 1] /= h
    boxes[:, 2] /= w
    boxes[:, 3] /= h
    return image_tensor, boxes


def resize_box(boxes, h, w):
    """

    :param boxes: Numpy.ndarray, (N, 4)
    :return:
    """
    xmin = boxes[:, 0] * h
    xmax = boxes[:, 1] * w
    ymin = boxes[:, 2] * h
    ymax = boxes[:, 3] * w
    boxes = np.stack([xmin, ymin, xmax, ymax], axis=1).astype(dtype=np.int32)
    return boxes
