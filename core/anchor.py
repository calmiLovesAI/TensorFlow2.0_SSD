import numpy as np
import math

from configuration import IMAGE_HEIGHT, IMAGE_WIDTH, ASPECT_RATIOS, FEATURE_MAPS, \
    DEFAULT_BOXES_SIZES, DOWNSAMPLING_RATIOS
from itertools import product


class DefaultBoxes:
    def __init__(self):
        assert IMAGE_HEIGHT == IMAGE_WIDTH
        self.image_size = IMAGE_HEIGHT
        self.num_priors = len(ASPECT_RATIOS)
        self.feature_maps = FEATURE_MAPS
        self.default_boxes_sizes = DEFAULT_BOXES_SIZES
        self.aspect_ratios = ASPECT_RATIOS
        self.steps = DOWNSAMPLING_RATIOS

    def generate_boxes(self):
        boxes = []
        for k, f in enumerate(self.feature_maps):
            for i, j in product(range(f[0]), repeat=2):
                f_k = self.image_size / self.steps[k]
                # box中心点的坐标
                center_x = (j + 0.5) / f_k
                center_y = (i + 0.5) / f_k
                # box的高和宽
                s_min = self.default_boxes_sizes[k][0] / self.image_size
                s_max = math.sqrt(self.default_boxes_sizes[k][0] * self.default_boxes_sizes[k][1]) / self.image_size
                boxes += [center_x, center_y, s_min, s_min]
                boxes += [center_x, center_y, s_max, s_max]
                for ar in self.aspect_ratios[k]:
                    boxes += [center_x, center_y, s_min * math.sqrt(ar), s_min / math.sqrt(ar)]

        output = np.array(boxes).reshape((-1, 4))
        output = np.clip(a=output, a_min=0, a_max=1)
        return output

