import tensorflow as tf

from configuration import IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS, EPOCHS, NUM_CLASSES, BATCH_SIZE, save_model_dir, \
    load_weights_before_training, load_weights_from_epoch, save_frequency
from core.ground_truth import ReadDataset, MakeGT
from core.loss import SSDLoss
from core.make_dataset import TFDataset
from core.ssd import SSD, ssd_prediction


def print_model_summary(network):
    network.build(input_shape=(None, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS))
    network.summary()


if __name__ == '__main__':
    # GPU settings
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    dataset = TFDataset()
    train_data, train_count = dataset.generate_datatset()

    ssd = SSD()
    print_model_summary(network=ssd)

    if load_weights_before_training:
        ssd.load_weights(filepath=save_model_dir+"epoch-{}".format(load_weights_from_epoch))
        print("Successfully load weights!")
    else:
        load_weights_from_epoch = -1

    # loss
    loss = SSDLoss()

    # optimizer
    optimizer = tf.optimizers.Adadelta()

    # metrics
    loss_metric = tf.metrics.Mean()

    def train_step(batch_images, batch_labels):
        with tf.GradientTape() as tape:
            pred = ssd(batch_images, training=True)
            output = ssd_prediction(feature_maps=pred, num_classes=NUM_CLASSES + 1)
            gt = MakeGT(batch_labels, pred)
            gt_boxes = gt.generate_gt_boxes()
            loss_value = loss(y_true=gt_boxes, y_pred=output)
        gradients = tape.gradient(loss_value, ssd.trainable_variables)
        optimizer.apply_gradients(grads_and_vars=zip(gradients, ssd.trainable_variables))
        loss_metric.update_state(values=loss_value)


    for epoch in range(load_weights_from_epoch + 1, EPOCHS):
        for step, batch_data in enumerate(train_data):
            images, labels = ReadDataset().read(batch_data)
            train_step(batch_images=images, batch_labels=labels)
            print("Epoch: {}/{}, step: {}/{}, loss: {:.9f}".format(epoch,
                                                                   EPOCHS,
                                                                   step,
                                                                   tf.math.ceil(train_count / BATCH_SIZE),
                                                                   loss_metric.result()))
        loss_metric.reset_states()

        if epoch % save_frequency == 0:
            ssd.save_weights(filepath=save_model_dir+"epoch-{}".format(epoch), save_format="tf")

    ssd.save_weights(filepath=save_model_dir+"saved_model", save_format="tf")
