import tensorflow as tf
import time

from configuration import IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS, EPOCHS, NUM_CLASSES, BATCH_SIZE, save_model_dir, \
    load_weights_from_epoch, save_frequency, test_images_during_training, \
    test_images_dir_list
from core.ground_truth import ReadDataset
from core.loss import MultiBoxLoss
from core.make_dataset import TFDataset
from core.ssd import SSD


from utils.visualize import visualize_training_results


def print_model_summary(network):
    sample_inputs = tf.random.normal(shape=(1, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS))
    _ = network(sample_inputs, training=True)
    network.summary()


def main():
    dataset = TFDataset()
    train_data, train_count = dataset.generate_datatset()

    model = SSD()
    print_model_summary(model)

    if load_weights_from_epoch >= 0:
        model.load_weights(filepath=save_model_dir + "epoch-{}".format(load_weights_from_epoch))
        print("成功从epoch-{}加载模型权重！".format(load_weights_from_epoch))

    loss_fn = MultiBoxLoss(num_classes=NUM_CLASSES, overlap_thresh=0.5, neg_pos=3)

    # optimizer
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-3,
                                                                 decay_steps=20000,
                                                                 decay_rate=0.96)
    optimizer = tf.optimizers.Adam(learning_rate=lr_schedule)

    loss_metric = tf.metrics.Mean()
    cls_loss_metric = tf.metrics.Mean()
    reg_loss_metric = tf.metrics.Mean()

    for epoch in range(load_weights_from_epoch + 1, EPOCHS):
        start_time = time.time()
        for step, batch_data in enumerate(train_data):
            images, labels = ReadDataset().read(batch_data)

            with tf.GradientTape() as tape:
                predictions = model(images, training=True)
                loss_l, loss_c = loss_fn(y_true=labels, y_pred=predictions)
                total_loss = loss_l + loss_c
            gradients = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(grads_and_vars=zip(gradients, model.trainable_variables))
            loss_metric.update_state(values=total_loss)
            cls_loss_metric.update_state(values=loss_c)
            reg_loss_metric.update_state(values=loss_l)

            time_per_step = (time.time() - start_time) / (step + 1)
            print("Epoch: {}/{}, step: {}/{}, speed: {:.2f}s/step, loss: {:.10f}, "
                  "cls loss: {:.10f}, reg loss: {:.10f}".format(epoch,
                                                                EPOCHS,
                                                                step,
                                                                tf.math.ceil(train_count / BATCH_SIZE),
                                                                time_per_step,
                                                                loss_metric.result(),
                                                                cls_loss_metric.result(),
                                                                reg_loss_metric.result()))
        loss_metric.reset_states()
        cls_loss_metric.reset_states()
        reg_loss_metric.reset_states()

        if epoch % save_frequency == 0:
            model.save_weights(filepath=save_model_dir + "epoch-{}".format(epoch), save_format="tf")

        if test_images_during_training:
            visualize_training_results(pictures=test_images_dir_list, model=model, epoch=epoch)

    model.save_weights(filepath=save_model_dir + "epoch-{}".format(EPOCHS), save_format="tf")


if __name__ == '__main__':
    print("TensoFlow版本：{}".format(tf.__version__))
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)
    main()


