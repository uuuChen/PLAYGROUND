import numpy as np
import random

from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class Data_Preprocessing:

    def __init__(self):
        pass

    def _flatten(self, arr):
        return arr.reshape(arr.shape[0], -1)

    def _one_hot(self, label, num_of_class):
        one_hot_label = np.eye(num_of_class)[label]
        return one_hot_label

    def get_MNIST_data_and_label(self):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.MNIST('./MNIST/', train=True, transform=transform, download=True)
        test_dataset = datasets.MNIST('./MNIST/', train=False, transform=transform, download=True)

        train_loader = DataLoader(train_dataset, batch_size=len(train_dataset))
        test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))

        train_data = next(iter(train_loader))[0].numpy()
        test_data = next(iter(test_loader))[0].numpy()

        train_label = self._one_hot(next(iter(train_loader))[1].numpy(), 10)
        test_label = self._one_hot(next(iter(test_loader))[1].numpy(), 10)

        return train_data, train_label, test_data, test_label

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









