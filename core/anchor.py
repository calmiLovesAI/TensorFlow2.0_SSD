import tensorflow as tf
import numpy as np

class Anchors():
    def __init__(self, feature_map, sizes, ratios):
        self.w = feature_map.shape[2]
        self.h = feature_map.shape[1]
        self.sizes = sizes
        self.ratios = ratios
        self.num_size = len(self.sizes)
        self.num_ratio = len(self.ratios)

    def __get_box_w_and_h(self, size, ratio):
        box_w = self.w * size * np.sqrt(ratio)
        box_h = self.h * size / np.sqrt(ratio)
        return box_w, box_h

    def __box_coordinate_normalization(self, x, y, box_info):
        box_info_array = np.array(box_info)
        box_w = box_info_array[:, 0]
        box_h = box_info_array[:, 1]
        x_min = x - 0.5 * box_w
        y_min = y - 0.5 * box_h
        x_max = x + 0.5 * box_w
        y_max = y + 0.5 * box_h

        # normalization
        x_min /= self.w
        x_max /= self.h
        y_min /= self.w
        y_max /= self.h

        num_of_boxes = len(box_info)
        coord_list = []
        for i in range(num_of_boxes):
            coord_list.append([x_min[i], y_min[i], x_max[i], y_max[i]])

        return tf.convert_to_tensor(coord_list)


    def __generate_anchors(self):
        # Sizes must belong to (0, 1], and ratios should be greater than 0.
        # Assuming that the number of sizes is n and the number of ratios is m,
        # n+m-1 anchor frames are generated at each pixel of the feature map.
        # The i-th anchor uses size[i] and ratios[0] (if i<=n); size[0] and ratios[i-n] (if i>n).

        box_info = []

        for i in range(self.num_size + self.num_ratio - 1):
            if i < self.num_size:
                box_w, box_h = self.__get_box_w_and_h(size=self.sizes[i],
                                                      ratio=self.ratios[0])
            else:
                box_w, box_h = self.__get_box_w_and_h(size=self.sizes[0],
                                                      ratio=self.ratios[i-self.num_size+1])
            box_info.append([box_w, box_h])

        return box_info

    def get_all_default_boxes(self):
        box_info = self.__generate_anchors()
        # initialize stack_ndarray
        stack_tensor = self.__box_coordinate_normalization(x=0, y=0, box_info=box_info)
        for x in range(self.w):
            for y in range(self.h):
                if x == 0 and y == 0:
                    stack_tensor = stack_tensor
                else:
                    # the shape of coordinates of each pixel on the feature map:[5, 4]
                    pixel_coord = self.__box_coordinate_normalization(x=x,
                                                                      y=y,
                                                                      box_info=box_info)
                    # stack_tensor = tf.stack((stack_tensor, pixel_coord), axis=2)
                    stack_tensor = tf.convert_to_tensor(np.dstack((stack_tensor.numpy(), pixel_coord.numpy())))

        boxes = tf.reshape(stack_tensor, (-1, 4, self.w, self.h))
        boxes = tf.transpose(boxes, perm=[2, 3, 0, 1])

        return boxes


