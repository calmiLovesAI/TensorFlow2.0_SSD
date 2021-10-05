# training parameters
EPOCHS = 50
BATCH_SIZE = 8

IMAGE_HEIGHT = 300
IMAGE_WIDTH = 300
CHANNELS = 3

load_weights_from_epoch = -1
save_frequency = 5

test_picture_dir = ""

test_images_during_training = False
training_results_save_dir = "./test_pictures/"
test_images_dir_list = ["", ""]

# When the iou value of the anchor and the real box is less than the IoU_threshold,
# the anchor is divided into negative classes, otherwise positive.
# IOU_THRESHOLD = 0.6

# generate anchor
ASPECT_RATIOS = [[2.0, 0.5],
                 [2.0, 0.5, 3.0, 1.0 / 3.0],
                 [2.0, 0.5, 3.0, 1.0 / 3.0],
                 [2.0, 0.5, 3.0, 1.0 / 3.0],
                 [2.0, 0.5],
                 [2.0, 0.5]]

# SSD中每个stage分支输出的feature map中每个像素位置处的先验框数量
STAGE_BOXES_PER_PIXEL = [len(x) + 2 for x in ASPECT_RATIOS]

DOWNSAMPLING_RATIOS = [8, 16, 32, 64, 100, 300]

# SSD网络结构的所有输出feature map的大小（H * W）
FEATURE_MAPS = [(38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1)]

# 每个feature map对应的先验框尺寸（相对于原始输入图片分辨率）
DEFAULT_BOXES_SIZES = [(30, 60), (60, 111), (111, 162), (162, 213), (213, 264), (264, 315)]

# focal loss
alpha = 0.25
gamma = 2.0

reg_loss_weight = 0.5

# dataset
PASCAL_VOC_DIR = "./dataset/VOCdevkit/VOC2012/"
# The 20 object classes of PASCAL VOC
# OBJECT_CLASSES = {"person": 1, "bird": 2, "cat": 3, "cow": 4, "dog": 5,
#                   "horse": 6, "sheep": 7, "aeroplane": 8, "bicycle": 9,
#                   "boat": 10, "bus": 11, "car": 12, "motorbike": 13,
#                   "train": 14, "bottle": 15, "chair": 16, "diningtable": 17,
#                   "pottedplant": 18, "sofa": 19, "tvmonitor": 20}

OBJECT_CLASSES = {"aeroplane": 0, "bicycle": 1, "bird": 2, "boat": 3,
                  "bottle": 4, "bus": 5, "car": 6, "cat": 7, "chair": 8,
                  "cow": 9, "diningtable": 10, "dog": 11, "horse": 12,
                  "motorbike": 13, "person": 14, "pottedplant": 15,
                  "sheep": 16, "sofa": 17, "train": 18, "tvmonitor": 19}

NUM_CLASSES = len(OBJECT_CLASSES) + 1

TXT_DIR = "voc.txt"


MAX_BOXES_PER_IMAGE = 20

VARIANCE = [0.1, 0.2]

# nms
NMS_THRESHOLD = 0.45
CONFIDENCE_THRESHOLD = 0.01
MAX_BOXES_NUM = 200


# directory of saving model
save_model_dir = "saved_model/"

