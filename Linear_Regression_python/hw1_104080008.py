from Data_Preprocessing import Data_Preprocessing
from Model import Model
from Parameters import  Parameters
import numpy as np
import os

TRAIN_TEST_RATIO = 0.8


class Core:

    def __init__(self, data_label_list):
        self.model_dict = {}
        [self.train_data, self.train_label, self.valid_data, self.valid_label] = data_label_list

    def process(self, type):
        parameters = Parameters()
        pars = parameters.get_pars(type=type)

        model = Model(pars=pars)
        model.train(X=self.train_data, y=self.train_label)

        RMSE = model.get_RMSE(X=self.valid_data, y=self.valid_label)
        model.console(RMSE=RMSE)
        self.model_dict[type] = model


def main():
    types = ['lr', 'lr_reg', 'lr_reg_bias', 'blr_bias']
    model_dict = {}

    # data_preprocess
    dp_train = Data_Preprocessing('train.csv', True, split_ratio=0.8)
    dp_test = Data_Preprocessing('test_no_G3.csv', False)

    train_data, train_label = dp_train.get_train_data_label()
    valid_data, valid_label = dp_train.get_valid_data_label()
    test_data = dp_test.get_test_data()

    core = Core([train_data, train_label, valid_data, valid_label])

    print('-----------------------------------------------------------------------------------')
    print('Linear Regression\n')

    for type in types:
        core.process(type=type)

    file_name = '104080008_1.txt'
    if not os.path.isfile(file_name):
        model_dict['lr_rb'].write_yhat_to_file(file_name, test_data)

    # Classification
    train_data, train_binary_label = dp_train.get_train_data_label(binary_label=True, threshold=10.0)
    valid_data, valid_binary_label = dp_train.get_valid_data_label(binary_label=True, threshold=10.0)

    core_binary = Core([train_data, train_binary_label, valid_data, valid_binary_label])

    print('-----------------------------------------------------------------------------------')
    print('Classification\n')

    types = ['lr_reg_bias_binary', 'LR_bias_binary']

    for type in types:
        core_binary.process(type)

    yhat_thresholds = [0.1, 0.5, 0.9]
    for type in types:
        print('')
        model = core_binary.model_dict[type]
        for yhat_threshold in yhat_thresholds:
            model.set_yhat_threshold(yhat_threshold)
            RMSE = model.get_RMSE(X=train_data, y=train_binary_label)
            model.console(RMSE=RMSE)
            c_m = model.get_confusion_matrix(X=train_data, y=train_binary_label)
            if (c_m[1][0] + c_m[1][1]) != 0:
                precision = c_m[1][1] / (c_m[1][0] + c_m[1][1])
            else:
                precision = 0
            accuracy = (c_m[0][0] + c_m[1][1]) / len(train_binary_label)
            print(c_m)
            print('accuracy : %s \t precision : %s' % (accuracy, precision))

    # predict test set and store in file
    test_data = dp_test.get_test_data()
    model = core_binary.model_dict['lr_reg_bias_binary']
    model.set_binary_label(False)

    file_name = '104080008_2.txt'
    if not os.path.isfile(file_name):
        model.write_yhat_to_file(file_name=file_name, X=test_data)


if __name__ == '__main__':
    main()















