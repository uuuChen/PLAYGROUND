import numpy as np


class Fully_Connected_Layer:

    def __init__(self, weight, bias):
        self.weight = weight
        self.bias = bias

    def forward_propagation(self, input_layer):
        self.input_layer = input_layer
        return np.matmul(self.input_layer, self.weight) + self.bias

    def backward_propagation(self, dLoss_dOutput):
        self.dLoss_dInput_layer = np.matmul(dLoss_dOutput, self.weight.T)
        self.dLoss_dWeight = np.matmul(self.input_layer.T, dLoss_dOutput)
        self.dLoss_dBias = np.mean(dLoss_dOutput, axis=0)
        return self.dLoss_dInput_layer

    def update_weight_and_bias(self, learning_rate):
        self.weight -= learning_rate * self.dLoss_dWeight
        self.bias -= learning_rate * self.dLoss_dBias


class Relu_Layer:

    def forward_propagation(self, input_layer):
        self.input_layer = input_layer
        return np.where(self.input_layer >= 0, self.input_layer, 0)

    def backward_propagation(self, dLoss_dOutput):
        dLoss_dInput = np.where(self.input_layer >= 0, dLoss_dOutput, 0)
        return dLoss_dInput


class Dropout_Layer:

    def forward_propagation(self, input_layer, keep_rate):
        self.keep_rate = keep_rate
        self.mask = [np.random.uniform(size=input_layer.shape[1]) <= self.keep_rate]
        return np.where(self.mask, input_layer / self.keep_rate, 0)

    def backward_propagation(self, dLoss_dOutput):
        return np.where(self.mask, dLoss_dOutput / self.keep_rate, 0)


class Sigmoid_Layer:

    def forward_propagation(self, input_layer):
        self.output = 1 / (1 + np.exp(-input_layer))
        return self.output

    def backward_propagation(self, dLoss_dOutput):
        dOutput_dInput = np.multiply(self.output, (1 - self.output))
        dLoss_dInput = np.multiply(dLoss_dOutput, dOutput_dInput)
        return dLoss_dInput


class Softmax_Layer:

    def forward_propagation(self, input_layer):
        self.input_layer = np.array([input_layer[row, :] - np.max(input_layer[row, :])
                                     for row in range(input_layer.shape[0])])
        self.output = np.zeros(shape=self.input_layer.shape)
        exp_input_layer = np.exp(self.input_layer)
        for row in range(self.output.shape[0]):
            self.output[row, :] = exp_input_layer[row, :] / np.sum(exp_input_layer[row, :])
        return self.output

    def backward_propagation(self, dLoss_dOutput):

        dOutput_dInput = np.zeros(shape=(self.input_layer.shape[0], self.input_layer.shape[1],
                                         self.input_layer.shape[1]))

        for row in range(dOutput_dInput.shape[0]):
            for i in range(dOutput_dInput.shape[1]):
                for j in range(dOutput_dInput.shape[2]):
                    if i == j:
                        dOutput_dInput[row, i, j] = self.output[row, i] * (1 - self.output[row, i])
                    else:
                        dOutput_dInput[row, i, j] = - self.output[row, i] * self.output[row, j]

        dLoss_dInput = np.zeros(shape=self.input_layer.shape)

        for row in range(dLoss_dInput.shape[0]):
            dLoss_dInput[row, :] = np.matmul(dLoss_dOutput[row, :], dOutput_dInput[row, :, :])
        return dLoss_dInput


class Cross_Entropy_Layer:

    def forward_propagation(self, predict, one_hot_label):
        self.predict = predict
        self.one_hot_label = one_hot_label
        self.epison = 1e-10

        self.output = np.mean(
                          np.sum(
                              - self.one_hot_label * np.log(self.predict + self.epison)
                              - (1 - self.one_hot_label) * np.log((1 - self.predict + self.epison)),
                              axis=1)
                       )
        return self.output

    def backward_propagation(self):

        dLoss_dPredict = (1 / self.one_hot_label.shape[0]) * \
                            ((- self.one_hot_label / (self.predict + self.epison)) +
                             ((1 - self.one_hot_label) / (1 - self.predict + self.epison)))

        return dLoss_dPredict






