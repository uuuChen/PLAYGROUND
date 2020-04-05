from Layers import *
from Data_Preprocessing import Data_Preprocessing
from NetWork import NetWork
from Optimizer import *


class CNN_Model(NetWork):

    def __init__(self, input_layer_shape, num_of_class):
        super(CNN_Model, self).__init__()
        self.global_step = 0
        self.graph = self.build_graph(input_layer_shape, num_of_class)

    def build_graph(self, input_layer_shape, num_of_class):
        graph = self.sequence(
                    Convolution_Layer(np.random.normal(size=(6, input_layer_shape[1], 5, 5)),
                                      np.zeros(shape=6), 1, 'VALID'),
                    Max_Pool_Layer([1, 1, 2, 2], 2, 'VALID'),
                    # Relu_Layer(),
                    Sigmoid_Layer(),

                    Convolution_Layer(np.random.normal(size=(16, 6, 5, 5)), np.zeros(shape=16), 1, 'VALID'),
                    Max_Pool_Layer([1, 1, 2, 2], 2, 'VALID'),
                    # Relu_Layer(),
                    Sigmoid_Layer(),

                    Flatten_Layer(),

                    Fully_Connected_Layer(np.random.normal(size=(16 * 4 * 4, 120)), np.zeros(shape=120)),
                    Sigmoid_Layer(),
                    # Dropout_Layer(),

                    Fully_Connected_Layer(np.random.normal(size=(120, 84)), np.zeros(shape=84)),
                    Sigmoid_Layer(),
                    # Dropout_Layer(),

                    Fully_Connected_Layer(np.random.normal(size=(84, num_of_class)), np.zeros(shape=num_of_class)),
                    Softmax_Layer(),
                    Cross_Entropy_Layer()
                )
        return graph

    def fit(self, epochs,  batch_size, keep_rate,  learning_rate,  train_data, train_label, test_data, test_label):
        train_batch_pointer = 0
        batches = train_data.shape[0] // batch_size + 1
        train_data, train_label = Data_Preprocessing.shuffle_data_label(train_data, train_label)
        for epoch in range(1, epochs + 1):
            self.global_step += 1
            for batch in range(1, batches + 1):
                train_batch_data, train_batch_label, train_batch_pointer = Data_Preprocessing.get_batch(train_data,
                                                                                                        train_label,
                                                                                                        train_batch_pointer,
                                                                                                        batch_size)

                learning_rate = Optimizer.learning_rate_exponential_decay(learning_rate, 0.95, self.global_step, 128)
                train_predict, train_loss = self.graph.train(train_batch_data, train_batch_label,
                                                             learning_rate=learning_rate, keep_rate=keep_rate)
                train_accuracy = self.get_accuracy(train_batch_label, train_predict)
                print(f'epoch: [{epoch}][{batch}/{batches}]\t'
                      f'loss: {train_loss:<.2f}\t'
                      f'accuracy: {train_accuracy:<.2f}')

            test_predict, test_loss = self.predict(test_data, test_label)
            test_accuracy = self.get_accuracy(test_label, test_predict)
            print(f'\nepoch-test: [{epoch}]\t'
                  f'loss: {test_loss:<.2f}\t'
                  f'acc: {test_accuracy:<.2f}\n')

    def get_accuracy(self, label, label_hat):
        return np.mean(np.argmax(label, axis=1) == np.argmax(label_hat, axis=1))

    def get_label_hat(self, predict):
        label_hat = np.zeros(shape=predict.shape)
        for row, max_idx in enumerate(np.argmax(predict, axis=1)):
            label_hat[row, max_idx] = 1
        return label_hat


if __name__ == '__main__':
    train_data, train_label, test_data, test_label = Data_Preprocessing().get_MNIST_data_and_label()
    model = CNN_Model([None, 1, 28, 28], 10)
    model.fit(300, 128, 0.5, 0.1, train_data, train_label, test_data, test_label)





