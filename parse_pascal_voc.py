import xml.dom.minidom as xdom
from configuration import PASCAL_VOC_DIR, OBJECT_CLASSES, IMAGE_WIDTH, IMAGE_HEIGHT
import os
import tensorflow as tf
from PIL import Image
from preprocess import preprocess_image

class ParsePascalVOC():
    def __init__(self):
        super(ParsePascalVOC, self).__init__()
        self.all_xml_dir = PASCAL_VOC_DIR + "Annotations"
        self.all_image_dir = PASCAL_VOC_DIR + "JPEGImages"

    # parse one xml file
    def __parse_xml(self, xml):
        obj_and_box_list = []
        DOMTree = xdom.parse(xml)
        annotation = DOMTree.documentElement
        image_name = annotation.getElementsByTagName("filename")[0].childNodes[0].data

        obj = annotation.getElementsByTagName("object")
        for o in obj:
            o_list = []
            obj_name = o.getElementsByTagName("name")[0].childNodes[0].data
            o_list.append(OBJECT_CLASSES[obj_name])
            bndbox = o.getElementsByTagName("bndbox")
            for box in bndbox:
                xmin = box.getElementsByTagName("xmin")[0].childNodes[0].data
                ymin = box.getElementsByTagName("ymin")[0].childNodes[0].data
                xmax = box.getElementsByTagName("xmax")[0].childNodes[0].data
                ymax = box.getElementsByTagName("ymax")[0].childNodes[0].data
                o_list.append(xmin)
                o_list.append(ymin)
                o_list.append(xmax)
                o_list.append(ymax)
                break
            obj_and_box_list.append(o_list)
        return image_name, obj_and_box_list

    def __prepare_dataset(self):
        image_path_list = []
        all_boxes_list = []
        for item in os.listdir(self.all_xml_dir):
            item_dir = os.path.join(self.all_xml_dir, item)
            image_name, boxes_list = self.__parse_xml(xml=item_dir)
            image_path_list.append(os.path.join(self.all_image_dir, image_name))
            all_boxes_list.append(boxes_list)

        # image_name : [picture_1_name, picture_2_name, ...]
        # all_boxes_list : [
        #                   [[obj_1's class, xmin, ymin, xmax, ymax], [obj_2's class, xmin, ymin, xmax, ymax], ...],
        #                   [[obj_1's class, xmin, ymin, xmax, ymax], [obj_2's class, xmin, ymin, xmax, ymax], ...],
        #                   ]
        return image_path_list, all_boxes_list

    def __scale_label(self, label, w_scale, h_scale):
        for item in label:
            # convert to float
            item[1] = float(item[1])
            item[2] = float(item[2])
            item[3] = float(item[3])
            item[4] = float(item[4])
            # rescale the coordinates' value
            item[1] *= w_scale
            item[2] *= h_scale
            item[3] *= w_scale
            item[4] *= h_scale

        return label

    def __get_labels(self, labels, image_path):
        label_list = []
        for i in range(len(labels)):
            # print(image_path[i])
            img = Image.open(image_path[i])
            w = img.size[0]
            h = img.size[1]
            w_scale = IMAGE_WIDTH / w
            h_scale = IMAGE_HEIGHT / h
            l = self.__scale_label(label=labels[i], w_scale=w_scale, h_scale=h_scale)
            label_list.append(l)
        return label_list

    def split_dataset(self):
        image_path, boxes = self.__prepare_dataset()

        labels = self.__get_labels(labels=boxes, image_path=image_path)
        labels = tf.convert_to_tensor(labels)
        print(labels)
        print(type(labels))

        image_dataset = tf.data.Dataset.from_tensor_slices(image_path).map(preprocess_image)
        label_dataset = tf.data.Dataset.from_tensor_slices(labels)


if __name__ == '__main__':
    parse = ParsePascalVOC()
    parse.split_dataset()