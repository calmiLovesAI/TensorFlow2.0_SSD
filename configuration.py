# training parameters
BATCH_SIZE = 1
NUM_CLASSES = 1
IMAGE_HEIGHT = 256
IAMGE_WIDTH = 256

# When the iou value of the anchor and the real box is less than the IoU_threshold,
# the anchor is divided into negative classes, otherwise positive.
IoU_threshold = 0.6

background_sample_num = 128

# focal loss
alpha = 0.25
gamma = 2

# dataset
PASCAL_VOC_DIR = "./dataset/VOCdevkit/VOC2012/"