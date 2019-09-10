# training parameters
BATCH_SIZE = 8
NUM_CLASSES = 1


# When the iou value of the anchor and the real box is less than the IoU_threshold,
# the anchor is divided into negative classes, otherwise positive.
IoU_threshold = 0.3