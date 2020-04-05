import numpy as np
import random
from tensorflow.examples.tutorials.mnist import input_data


class Data_Preprocessing:

    def __init__(self):
        self.mnist = input_data.read_data_sets('MNIST', one_hot=True)

    def get_data_and_label(self):
        train_data = np.reshape(self.mnist.train.images, [-1, 1, 28, 28])
        train_label = self.mnist.train.labels
        test_data = np.reshape(self.mnist.test.images, [-1, 1, 28, 28])
        test_label = self.mnist.test.labels
        return [train_data, train_label, test_data, test_label]

    @staticmethod
    def get_batch(data, label,batch_pointer, batch_size):
        row_start = batch_pointer
        row_end = row_start + batch_size - 1
        batch_pointer += batch_size
        if row_end >= (len(data) - 1):
            row_end = len(data) - 1
            batch_pointer = 0
        batch_data = data[row_start: row_end + 1]
        batch_label = label[row_start: row_end + 1]
        return batch_data, batch_label, batch_pointer

    @staticmethod
    def shuffle_data_label(data, label):
        data_label = list(zip(data, label))
        random.shuffle(data_label)
        data, label = zip(*data_label)
        return np.array(data), np.array(label)


if __name__ == '__main__':
    dp = Data_Preprocessing()
    data_label = dp.get_data_and_label()








