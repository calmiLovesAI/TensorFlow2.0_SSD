from xml.dom.minidom import parse
from configuration import PASCAL_VOC_DIR
import os

# parse one xml file
def parse_xml(xml):
    DOMTree = parse(xml)
    annotation = DOMTree.documentElement
    image_name = annotation.getElementsByTagName("filename")[0]
    print("image_name : ", image_name.childNodes[0].data)

    obj = annotation.getElementsByTagName("object")
    for o in obj:
        obj_name = o.getElementsByTagName("name")[0]
        print("object name : ", obj_name.childNodes[0].data)
        bndbox = o.getElementsByTagName("bndbox")[0]


if __name__ == '__main__':
    all_xml_dir = PASCAL_VOC_DIR + "Annotations"
    for item in os.listdir(all_xml_dir):
        item_dir = os.path.join(all_xml_dir, item)
        parse_xml(xml=item_dir)