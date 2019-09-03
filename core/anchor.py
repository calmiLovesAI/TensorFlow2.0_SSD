import tensorflow as tf
import numpy as np

class Anchors():
    def __init__(self, feature_map_w, feature_map_h, sizes, ratios):
        self.w = feature_map_w
        self.h = feature_map_h
        self.sizes = sizes
        self.ratios = ratios
        self.num_size = len(self.sizes)
        self.num_ratio = len(self.ratios)

    def __get_box_w_and_h(self, size, ratio):
        box_w = self.w * size * np.sqrt(ratio)
        box_h = self.h * size / np.sqrt(ratio)
        return box_w, box_h

    def __get_all_pixel_coordinate(self):
        x = np.arange(self.w)
        y = np.arange(self.h)
        x_m, y_m = np.meshgrid(x, y)
        x_m = x_m.flatten()
        y_m = y_m.flatten()
        return zip(x_m, y_m)

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
        return x_min, y_min, x_max, y_max


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
        pixel_coord = self.__get_all_pixel_coordinate()
        box_info = self.__generate_anchors()
        for item in pixel_coord:
            x_min, y_min, x_max, y_max = self.__box_coordinate_normalization(x=item[0],
                                                                             y=item[1],
                                                                             box_info=box_info)


