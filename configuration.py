# training parameters
EPOCHS = 50
BATCH_SIZE = 8
NUM_CLASSES = 20
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
CHANNELS = 3

# When the iou value of the anchor and the real box is less than the IoU_threshold,
# the anchor is divided into negative classes, otherwise positive.
IoU_threshold = 0.6

background_sample_num = 128

# focal loss
alpha = 0.25
gamma = 2

# dataset
PASCAL_VOC_DIR = "./dataset/VOCdevkit/VOC2012/"
# The 20 object classes of PASCAL VOC
OBJECT_CLASSES = {"person": 1, "bird": 2, "cat": 3, "cow": 4, "dog": 5,
                  "horse": 6, "sheep": 7, "aeroplane": 8, "bicycle": 9,
                  "boat": 10, "bus": 11, "car": 12, "motorbike": 13,
                  "train": 14, "bottle": 15, "chair": 16, "diningtable": 17,
                  "pottedplant": 18, "sofa": 19, "tvmonitor": 20}
# (0, 1)
train_ratio = 0.8
