from Layers import *
from Data_Preprocessing import Data_Preprocessing

class MLP_Model:

    def __init__(self,
                 n_features,
                 n_hiddens,
                 n_labels,
                 activation_func,
                 learning_rate):

        self.n_features = n_features
        self.n_hiddens = n_hiddens
        self.n_labels = n_labels
        self.layers_units = [n_features] + n_hiddens + [n_labels]
        self.activation_func = activation_func
        self.learning_rate = learning_rate

        self.weights, self.biases = self.init_weights_and_biases(layers_units=self.layers_units)

        self.layers = self.init_structure(layers_units=self.layers_units,
                                          activation_func=self.activation_func,
                                          weights=self.weights,
                                          biases=self.biases)

    def init_weights_and_biases(self,
                                layers_units):

        weights = {}
        biases = {}

        for layer in range(1, len(layers_units)):
            fc_key = 'fc' + str(layer)
            weights[fc_key] = np.random.randn(layers_units[layer - 1], layers_units[layer])
            biases[fc_key] = np.zeros(layers_units[layer])

        return weights, biases

    def init_structure(self,
                       layers_units,
                       activation_func,
                       weights,
                       biases):

        layers = []

        for layer in range(1, len(layers_units)):

            fc_key = 'fc' + str(layer)

            layers.append(
                Fully_Connected_Layer(weight=weights[fc_key],
                                      bias=biases[fc_key])
            )

            if layer != len(layers_units) - 1:

                if activation_func == 'RELU':
                    layers.append(
                        Relu_Layer()
                    )

                elif activation_func == 'SIGMOID':
                    layers.append(
                        Sigmoid_Layer()
                    )

                layers.append(
                    Dropout_Layer()
                )

        layers.append(
            Softmax_Layer()
        )

        layers.append(
            Cross_Entropy_Layer()
        )

        return layers


    def fit(self,
            epochs,
            batch_size,
            keep_rate,
            train_data,
            train_label,
            test_data,
            test_label):

        train_batch_pointer = 0

        batches = train_data.shape[0] // batch_size + 1

        for epoch in range(1, epochs + 1):

            train_data, train_label = Data_Preprocessing.shuffle_data_label(data=train_data,
                                                                            label=train_label)

            test_data, test_label = Data_Preprocessing.shuffle_data_label(data=test_data,
                                                                          label=test_label)

            for batch in range(1, batches + 1):

                train_batch_data, train_batch_label, train_batch_pointer = Data_Preprocessing.get_batch(data=train_data,
                                                                                                        label=train_label,
                                                                                                        batch_size=batch_size,
                                                                                                        batch_pointer=train_batch_pointer)
                train_label_hat, train_loss = self.forward_propagation(layers=self.layers,
                                                                       input_data=train_batch_data,
                                                                       one_hot_label=train_batch_label,
                                                                       keep_rate=keep_rate)

                train_accuracy = self.get_accuracy(label=train_batch_label, label_hat=train_label_hat)

                self.backward_propagation(layers=self.layers)

                self.update_weights_and_biases(layers=self.layers,
                                               learning_rate=self.learning_rate)

            test_label_hat, test_loss = self.forward_propagation(layers=self.layers,
                                                                 input_data=test_data,
                                                                 one_hot_label=test_label,
                                                                 keep_rate=1.0)

            test_accuracy = self.get_accuracy(label=test_label, label_hat=test_label_hat)

            print('epoch ' + str(epoch))

            print('train loss %5s \t train accuracy %5s' % (np.round(train_loss, 2), np.round(train_accuracy, 2)))

            print('test loss %5s \t test accuracy %5s' % (np.round(test_loss, 2), np.round(test_accuracy, 2)))

    def forward_propagation(self,
                            layers,
                            input_data,
                            keep_rate,
                            one_hot_label):

        next_layer_input = input_data

        for layer in layers:

            if type(layer) == Softmax_Layer:
                predict = layer.forward_propagation(input_layer=next_layer_input)

            elif type(layer) == Cross_Entropy_Layer:
                loss = layer.forward_propagation(predict=predict,
                                                 one_hot_label=one_hot_label)
            elif type(layer) == Dropout_Layer:
                next_layer_input = layer.forward_propagation(input_layer=next_layer_input,
                                                             keep_rate=keep_rate)
            else:
                next_layer_input = layer.forward_propagation(input_layer=next_layer_input)

        label_hat = self.get_label_hat(predict=predict)

        return label_hat, loss

    def backward_propagation(self,
                             layers):

        reverse_layers = layers.copy()
        reverse_layers.reverse()
        for layer in reverse_layers:
            if type(layer) == Cross_Entropy_Layer:
                dLoss_dOutput = layer.backward_propagation()

            else:
                dLoss_dOutput = layer.backward_propagation(dLoss_dOutput=dLoss_dOutput)


    def update_weights_and_biases(self,
                                  layers,
                                  learning_rate):

        for layer in layers:
            if type(layer) == Fully_Connected_Layer:
                layer.update_weight_and_bias(learning_rate=learning_rate)


    def get_label_hat(self,
                      predict):

        label_hat = np.zeros(shape=predict.shape)
        for row, max_idx in enumerate(np.argmax(predict, axis=1)):
            label_hat[row, max_idx] = 1

        return label_hat

    def get_accuracy(self,
                     label,
                     label_hat):

        return np.mean(
            np.argmax(label, axis=1) == np.argmax(label_hat, axis=1)
        )

    def predict(self,
                input_data,
                one_hot_label):

        label_hat, loss = self.forward_propagation(layers=self.layers,
                                                   input_data=input_data,
                                                   one_hot_label=one_hot_label,
                                                   keep_rate=1.0)

        return label_hat, loss





