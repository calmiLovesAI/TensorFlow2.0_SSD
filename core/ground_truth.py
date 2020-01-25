import tensorflow as tf
import numpy as np
from utils.tools import str_to_int, resize_box
from configuration import MAX_BOXES_PER_IMAGE


class ReadDataset(object):
    def __init__(self):
        pass

    @staticmethod
    def __get_image_information(single_line):
        line_string = bytes.decode(single_line.numpy(), encoding="utf-8")
        line_list = line_string.strip().split(" ")
        image_name, image_height, image_width = line_list[:3]
        image_height, image_width = str_to_int(image_height), str_to_int(image_width)
        boxes = []
        num_of_boxes = (len(line_list) - 3) / 5
        if int(num_of_boxes) == num_of_boxes:
            num_of_boxes = int(num_of_boxes)
        else:
            raise ValueError("num_of_boxes must be 'int'.")
        for index in range(num_of_boxes):
            if index < MAX_BOXES_PER_IMAGE:
                xmin = str_to_int(line_list[3 + index * 5])
                ymin = str_to_int(line_list[3 + index * 5 + 1])
                xmax = str_to_int(line_list[3 + index * 5 + 2])
                ymax = str_to_int(line_list[3 + index * 5 + 3])
                class_id = int(line_list[3 + index * 5 + 4])
                xmin, ymin, xmax, ymax = resize_box(image_height, image_width, xmin, ymin, xmax, ymax)
                boxes.append([xmin, ymin, xmax, ymax, class_id])
        num_padding_boxes = MAX_BOXES_PER_IMAGE - num_of_boxes
        if num_padding_boxes > 0:
            for i in range(num_padding_boxes):
                boxes.append([0, 0, 0, 0, -1])
        boxes_array = np.array(boxes, dtype=np.float32)  # shape: (MAX_BOXES_PER_IMAGE, 5)
        return image_name, boxes_array

    def read(self, batch_data):
        image_name_list = []
        boxes_list = []
        for item in range(batch_data.shape[0]):
            image_name, boxes = self.__get_image_information(single_line=batch_data[item])
            image_name_list.append(image_name)
            boxes_list.append(boxes)
        boxes = np.stack(boxes_list, axis=0)   # shape : (batch_size, MAX_BOXES_PER_IMAGE, 5)
        return image_name_list, boxes


class MakeGT(object):
    def __init__(self):
        pass