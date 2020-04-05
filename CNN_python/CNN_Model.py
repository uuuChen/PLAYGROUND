from Layers import *
from Data_Preprocessing import Data_Preprocessing
from NetWork import NetWork
from Optimizer import *


class CNN_Model(NetWork):

    def __init__(self, input_layer_shape):
        super(CNN_Model, self).__init__()
        self.global_step = 0
        self.graph = self.build_graph(input_layer_shape=input_layer_shape)

    def build_graph(self, input_layer_shape):
        graph = self.sequence(
                    Convolution_Layer(np.random.normal(size=(6, input_layer_shape[1], 5, 5)),
                                      np.zeros(shape=6), strides=1, padding='VALID'),
                    Max_Pool_Layer([1, 1, 2, 2], 2, 'VALID'),
                    Relu_Layer(),
                    Convolution_Layer(np.random.normal(size=(16, 6, 5, 5)), np.zeros(shape=16), 1, 'VALID'),
                    Max_Pool_Layer([1, 1, 2, 2], 2, 'VALID'),
                    Relu_Layer(),
                    Flatten_Layer(),
                    Fully_Connected_Layer(np.random.normal(size=(16 * 4 * 4, 120)), np.zeros(shape=120)),
                    Sigmoid_Layer(),
                    # Dropout_Layer(),
                    Fully_Connected_Layer(np.random.normal(size=(120, 84)), np.zeros(shape=84)),
                    Sigmoid_Layer(),
                    # Dropout_Layer(),
                    Fully_Connected_Layer(np.random.normal(size=(84, 10)), np.zeros(shape=10)),
                    Softmax_Layer(),
                    Cross_Entropy_Layer()
                )
        return graph

    def fit(self, epochs,  batch_size, keep_rate,  learning_rate,  train_data, train_label, test_data, test_label):
        train_batch_pointer = 0
        batches = train_data.shape[0] // batch_size + 1
        for epoch in range(1, epochs + 1):
            train_data, train_label = Data_Preprocessing.shuffle_data_label(train_data, train_label)
            test_data, test_label = Data_Preprocessing.shuffle_data_label(test_data, test_label)
            for batch in range(1, batches + 1):
                train_batch_data, train_batch_label, train_batch_pointer = Data_Preprocessing.get_batch(train_data,
                                                                                                        train_label,
                                                                                                        batch_size,
                                                                                                        train_batch_pointer)

                learning_rate = Optimizer.learning_rate_exponential_decay(learning_rate, 0.95, self.global_step, 128)
                train_predict, train_loss = self.graph.train(train_batch_data, train_batch_label, learning_rate,
                                                             keep_rate)
                train_accuracy = self.get_accuracy(train_batch_label, train_predict)
                self.global_step += 1
                print('epoch ' + str(epoch))
                print('train loss %5s \t train accuracy %5s' % (np.round(train_loss, 2), np.round(train_accuracy, 2)))

            test_predict, test_loss = self.predict(test_data, test_label)
            test_accuracy = self.get_accuracy(test_label, test_predict)
            print('test loss %5s \t test accuracy %5s' % (np.round(test_loss, 2), np.round(test_accuracy, 2)))

    def get_accuracy(self, label, label_hat):
        return np.mean(np.argmax(label, axis=1) == np.argmax(label_hat, axis=1))

    def get_label_hat(self, predict):
        label_hat = np.zeros(shape=predict.shape)
        for row, max_idx in enumerate(np.argmax(predict, axis=1)):
            label_hat[row, max_idx] = 1
        return label_hat


if __name__ == '__main__':
    dp = Data_Preprocessing()
    train_data, train_label, test_data, test_label = dp.get_data_and_label()
    model = CNN_Model(input_layer_shape=[None, 1, 28, 28])
    model.fit(300, 128, 0.5, 0.01, train_data, train_label, test_data, test_label)





