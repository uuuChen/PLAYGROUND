import csv
import numpy as np


class Data_Preprocessing:

    PREDICTORS_TEST = np.array(
        ['ID', 'school', 'sex', 'age', 'famsize', 'studytime', 'failures', 'activities', 'higher',
         'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences'])
    PREDICTORS_TRAIN = np.append(PREDICTORS_TEST, 'G3')
    NO_NORNALIZE_PREDICTORS = np.array(['ID', 'G3'])
    BINARY_ATTRIBUTE_VALUE_DICT = {
        'yes': '1.0',
        'no':  '0.0',
        'GP':  '1.0',
        'MS':  '0.0',
        'F':   '1.0',
        'M':   '0.0',
        'LE3': '1.0',
        'GT3': '0.0'
    }

    def __init__(self, file, train, split_ratio=0.8):
        self.file = file
        self.train = train
        if train:
            self.split_ratio = split_ratio
        else:
            self.split_ratio = 1.0

        self.table = self.open_csv_file(self.file)
        self.predictor_table = self.init_predictor_table(self.table)
        self.transform_table = self.init_transform_table(self.predictor_table)
        self.normalized_table = self.init_normalized_table(self.transform_table)
        self.train_table, self.test_table = self.split_table(self.normalized_table)

    def open_csv_file(self, file_name):
        with open(file_name, newline='') as csvfile:
            rows = csv.reader(csvfile)
            table = []
            for row in rows:
                table.append(row)
            return np.array(table, dtype=object)

    def init_predictor_table(self, table):
        fields, records = table[0], table[1:]
        predictors = self.PREDICTORS_TRAIN if self.train else self.PREDICTORS_TEST

        # get a bool array of target features
        predictor_bool_list = [False] * len(fields)
        for predictor in predictors:
            idx = np.where(fields == predictor)[0][0]
            predictor_bool_list[idx] = True

        # reset table with target characteristics
        predictor_table = predictors
        num_records = len(records)
        for row in range(1, num_records + 1):
            predictor_table = np.vstack((predictor_table, table[row][predictor_bool_list]))
        return predictor_table

    def init_transform_table(self, table):
        fields, records = table[0], table[1:]
        record_row, record_col = len(records), len(records[0])
        for row in range(0, record_row):
            for col in range(0, record_col):
                key = records[row, col]
                if key in self.BINARY_ATTRIBUTE_VALUE_DICT.keys():
                    records[row, col] = self.BINARY_ATTRIBUTE_VALUE_DICT[key]
        records = records.astype(float)
        transform_table = np.vstack((fields, records))
        return transform_table

    def init_normalized_table(self, table):
        fields = table[0]
        records = table[1:]
        normalized_records = np.zeros(records.shape)
        for col in range(0, len(fields)):
            attributes = records[:, col]
            if fields[col] not in self.NO_NORNALIZE_PREDICTORS:
                mean = np.mean(attributes)
                std = np.std(attributes)
                normalized_attributes = (attributes - mean) / std
                normalized_records[:, col] = normalized_attributes
            else:
                normalized_records[:, col] = attributes
        normal_table = np.vstack((fields,  normalized_records))
        return normal_table

    def split_table(self, table):
        fields, records = table[0], table[1:]
        records_num = len(records)
        np.random.shuffle(records)
        train_num = int(records_num * self.split_ratio)
        train_table = np.insert(records[:train_num, :], 0, fields, axis=0)
        test_table = np.insert(records[train_num:, :], 0, fields, axis=0)
        return train_table, test_table

    def get_predictor_table(self):
        return self.predictor_table

    def get_transform_table(self):
        return self.transform_table

    def get_normalized_table(self):
        return self.normalized_table

    def get_train_table(self):
        return self.train_table

    def get_test_table(self):
        return self.test_table

    def get_train_data_label(self, binary_label=False, threshold=None):
        train_table = self.get_train_table()
        train_data, train_label = train_table[1:, 1:-1], train_table[1:, -1]
        if binary_label:
            train_label = np.where(train_label >= threshold, 1, 0)
        return train_data, train_label

    def get_valid_data_label(self, binary_label=False, threshold=None):
        valid_table = self.get_test_table()
        valid_data, valid_label = valid_table[1:, 1:-1], valid_table[1:, -1]
        if binary_label:
            valid_label = np.where(valid_label >= threshold, 1, 0)
        return valid_data, valid_label

    def get_test_data(self):
        test_table = self.get_train_table()
        return test_table[1:, 1:]




