import numpy as np
import os
import cv2
import random
from sklearn.decomposition import PCA


class Data_Preprocessing:

    def __init__(self, file_path, label_names, shuffle=False, data_process=False, one_hot_transform=False):
        self.file_path = file_path
        self.label_names = label_names
        self.batch_pointer = 0
        self.data, self.label = self.read_data_label_from_file(file_path)
        if shuffle:
            self.data, self.label = self.shuffle_data_label(self.data, self.label)
        if data_process:
            self.data = self.process_data(self.data)
        if one_hot_transform:
            self.label = self.one_hot_transform(self.label, len(label_names))

    def _write_data_path_to_file(self, data_dir, file_path):
        s = ''
        for dir_path, dir_names, file_names in os.walk(data_dir):
            for file_name in file_names:
                data_path = os.path.join(dir_path, file_name)
                label = file_name.split('_')[0]
                s += data_path + ' ' + label + '\n'
        file = open(file_path, 'w')
        file.write(s)
        file.close()

    def read_data_label_from_file(self, file_path):
        data_list = []
        label_list = []
        file = open(file_path, "r")
        for line in file:
            file_path = line.split()[0]
            label = line.split()[1]
            rgb_data = cv2.imread(file_path)
            gray_data = cv2.cvtColor(rgb_data, cv2.COLOR_RGB2GRAY)
            data_list.append(rgb_data)
            label_list.append(label)
        file.close()
        return np.array(data_list)/255, np.array(label_list)

    def process_data(self, data):
        flatten_data = np.reshape(data, (data.shape[0], -1))
        pca_data = self.get_PCA_data(data=flatten_data, n_components=2)
        return pca_data

    def one_hot_transform(self, label, n_labels):
        one_hot = np.zeros((label.shape[0], n_labels))
        label_index_dict = dict(zip(self.label_names, [i for i in range(n_labels)]))
        for row in range(len(label)):
            one_hot[row, label_index_dict[label[row]]] = 1
        return one_hot

    def get_data_and_label(self):
        data, label = np.reshape(self.data, (self.data.shape[0], -1)), self.label
        return data, label

    @staticmethod
    def get_PCA_data(data, n_components):
        pca = PCA(n_components=n_components, copy=True)
        return pca.fit_transform(data)

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








