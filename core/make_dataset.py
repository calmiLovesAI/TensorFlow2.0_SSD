import tensorflow as tf
from configuration import TXT_DIR, BATCH_SIZE


class TFDataset:
    def __init__(self):
        self.txt_dir = TXT_DIR

    @staticmethod
    def get_length_of_dataset(dataset):
        count = 0
        for _ in dataset:
            count += 1
        return count

    def generate_datatset(self):
        dataset = tf.data.TextLineDataset(filenames=self.txt_dir)
        length_of_dataset = self.get_length_of_dataset(dataset)
        train_dataset = dataset.batch(batch_size=BATCH_SIZE).shuffle(buffer_size=1000)
        return train_dataset, length_of_dataset
