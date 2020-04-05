from Data_Preprocessing import Data_Preprocessing
from MLP_Model import MLP_Model

TRAIN_FILE_PATH = 'train_file_path.txt'
TEST_FILE_PATH = 'test_file_path.txt'
LABEL_NAMES = ['Carambula', 'Lychee', 'Pear']
EPOCHS = 4000
FULL_BATCH = 1470
BATCH_SIZE = 128
N_HIDDENS = [256]
LEARNING_RATE = 0.01
KEEP_RATE = 0.5
# ACTIVATION_FUNC = 'RELU'
ACTIVATION_FUNC = 'SIGMOID'


def main():

    train_dp = Data_Preprocessing(TRAIN_FILE_PATH, LABEL_NAMES, shuffle=True, data_process=False, one_hot_transform=True)
    test_dp = Data_Preprocessing(TEST_FILE_PATH, LABEL_NAMES, shuffle=False, data_process=False, one_hot_transform=True)

    train_data, train_label = train_dp.get_data_and_label()
    test_data, test_label = test_dp.get_data_and_label()

    model = MLP_Model(train_data.shape[1], N_HIDDENS, len(LABEL_NAMES), ACTIVATION_FUNC, LEARNING_RATE)
    model.fit(EPOCHS, BATCH_SIZE, KEEP_RATE, train_data, train_label, test_data, test_label)
    test_label_hat, _ = model.predict(test_data, test_label)


if __name__ == '__main__':
    main()









