from Layers import *


class NetWork:

    def __init__(self, layers=None):
        self.layers = [] if layers is None else layers

    def _forward_propagation(self, input_data, input_label, keep_rate):
        next_input_layer = input_data
        for layer in self.layers:
            if type(layer) == Softmax_Layer:
                predict = layer.forward_propagation(next_input_layer)
            elif type(layer) == Cross_Entropy_Layer:
                loss = layer.forward_propagation(predict, input_label)
            elif type(layer) == Dropout_Layer:
                next_input_layer = layer.forward_propagation(next_input_layer, keep_rate)
            else:
                next_input_layer = layer.forward_propagation(next_input_layer)
        return predict, loss

    def _backward_propagation(self):
        reverse_layers = self.layers.copy()
        reverse_layers.reverse()
        for layer in reverse_layers:
            if type(layer) == Cross_Entropy_Layer:
                dLoss_dOutput = layer.backward_propagation()
            else:
                dLoss_dOutput = layer.backward_propagation(dLoss_dOutput)

    def _update(self, learning_rate):
        for layer in self.layers:
            if type(layer) == Convolution_Layer:
                layer.update_filter_and_bias(learning_rate)
            elif type(layer) == Fully_Connected_Layer:
                layer.update_weight_and_bias(learning_rate)

    def train(self, input_data, input_label, learning_rate=0.001, keep_rate=1.0):
        assert self.layers != [], 'you haven\'t build the graph yet'
        predict, loss = self._forward_propagation(input_data, input_label, keep_rate)
        self._backward_propagation()
        self._update(learning_rate)
        return predict, loss

    def sequence(self, *layers):
        for layer in list(layers):
            self.layers.append(layer)
        return NetWork(self.layers)

    def predict(self, input_data, input_label):
        predict, loss = self._forward_propagation(input_data, input_label, 1.0)
        return predict, loss







