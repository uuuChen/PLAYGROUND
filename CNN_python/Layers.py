import numpy as np


class Fully_Connected_Layer:

    def __init__(self, weight,  bias):
        self.weight = weight
        self.bias = bias

    def forward_propagation(self, input_layer):
        self.input_layer = input_layer
        return np.matmul(self.input_layer, self.weight) + self.bias

    def backward_propagation(self, dLoss_dOutput):
        self.dLoss_dInput_layer = np.matmul(dLoss_dOutput, self.weight.T)
        self.dLoss_dWeight = np.matmul(self.input_layer.T, dLoss_dOutput)
        self.dLoss_dBias = np.sum(dLoss_dOutput, axis=0)
        return self.dLoss_dInput_layer

    def update_weight_and_bias(self, learning_rate):
        self.weight -= learning_rate * self.dLoss_dWeight
        self.bias -= learning_rate * self.dLoss_dBias


class Zero_Padding:

    def __init__(self):
        pass

    def zero_padding_with_stride(self, input_layer, strides):
        output = np.zeros((input_layer.shape[0],
                           input_layer.shape[1],
                           input_layer.shape[2] + (input_layer.shape[2] - 1) * (strides - 1),
                           input_layer.shape[3] + (input_layer.shape[3] - 1) * (strides - 1)))

        np.copyto(output[:, :, ::strides, ::strides], input_layer)

        return output

    def zero_padding_with_filter(self, input_layer, filter_shape):
        top_zero_rows = bottom_zero_rows = filter_shape[2] - 1
        left_zero_cols = right_zero_cols = filter_shape[3] - 1

        output = input_layer
        output = np.insert(output, [0] * top_zero_rows + [output.shape[2]] * bottom_zero_rows, 0, axis=2)
        output = np.insert(output, [0] * left_zero_cols + [output.shape[3]] * right_zero_cols, 0, axis=3)

        return output

    def input_layer_zero_padding(self, input_layer, filter_shape, strides):
        zero_rows = ((input_layer.shape[2] - 1) // strides) * strides + filter_shape[2] - input_layer.shape[2]
        zero_cols = ((input_layer.shape[3] - 1) // strides) * strides + filter_shape[3] - input_layer.shape[3]

        top_zero_rows = zero_rows // 2
        bottom_zero_rows = zero_rows - top_zero_rows

        left_zero_cols = zero_cols // 2
        right_zero_cols = zero_cols - left_zero_cols

        output = input_layer
        output = np.insert(output, [0] * top_zero_rows + [output.shape[2]] * bottom_zero_rows, 0, axis=2)
        output = np.insert(output, [0] * left_zero_cols + [output.shape[3]] * right_zero_cols, 0, axis=3)

        input_layer_origin_mask = np.zeros(shape=output.shape, dtype=bool)
        input_layer_origin_mask[
            :,
            :,
            top_zero_rows: top_zero_rows + input_layer.shape[2],
            left_zero_cols: left_zero_cols + input_layer.shape[3]
        ] = 1

        return output, input_layer_origin_mask

    def dLoss_dOutput_zero_padding(self, dLoss_dOutput, filter_shape, strides):
        output = self.zero_padding_with_stride(dLoss_dOutput, strides)
        output = self.zero_padding_with_filter(output, filter_shape)
        return output


class Convolution_Layer(Zero_Padding):

    def __init__(self, filter, bias, strides, padding):
        super(Convolution_Layer, self).__init__()
        self.filter = filter
        self.bias = bias
        self.strides = strides
        self.padding = padding

    def forward_propagation(self, input_layer):
        self.input_layer = input_layer
        if self.padding == 'SAME':
            self.input_layer_zp, self.input_layer_origin_mask = self.input_layer_zero_padding(self.input_layer,
                                                                                              self.filter.shape,
                                                                                              self.strides)
        input_layer = self.input_layer_zp if self.padding == 'SAME' else self.input_layer
        output = self.convolution(input_layer, self.filter, self.bias, strides=self.strides)
        return output

    def compute_dLoss_dInput(self, dLoss_dOutput):
        filter_rot180 = np.rot90(self.filter, k=2, axes=(2, 3))
        filter_rot180_T = np.transpose(filter_rot180, axes=(1, 0, 2, 3))
        dLoss_dOutput_zp = self.dLoss_dOutput_zero_padding(dLoss_dOutput, self.filter.shape, self.strides)
        dLoss_dInput = self.convolution(dLoss_dOutput_zp, filter_rot180_T, np.zeros(shape=self.filter.shape[0]), 1)
        if self.padding == 'SAME':
            dLoss_dInput = dLoss_dInput[self.input_layer_origin_mask]
            dLoss_dInput = np.reshape(dLoss_dInput, self.input_layer.shape)
        return dLoss_dInput

    def compute_dLoss_dFilter(self, dLoss_dOutput):
        input_layer = self.input_layer_zp if self.padding == 'SAME' else self.input_layer
        dLoss_dOutput_zp = self.zero_padding_with_stride(dLoss_dOutput, self.strides)
        input_layer_T = np.transpose(input_layer, axes=(1, 0, 2, 3))
        dLoss_dOutput_zp_T = np.transpose(dLoss_dOutput_zp, axes=(1, 0, 2, 3))
        dLoss_dFilter_T = self.convolution(input_layer_T, dLoss_dOutput_zp_T, np.zeros(shape=self.filter.shape[0]), 1)
        dLoss_dFilter = np.transpose(dLoss_dFilter_T, axes=(1, 0, 2, 3))
        return dLoss_dFilter

    def compute_dLoss_dBias(self, dLoss_dOutput):
        dLoss_dBias = np.zeros(shape=self.bias.shape)
        for ch in range(dLoss_dOutput.shape[1]):
            dLoss_dBias[ch] = np.sum(dLoss_dOutput[:, ch, :, :])
        return dLoss_dBias

    def backward_propagation(self, dLoss_dOutput):
        self.dLoss_dInput = self.compute_dLoss_dInput(dLoss_dOutput)
        self.dLoss_dFilter = self.compute_dLoss_dFilter(dLoss_dOutput)
        self.dLoss_dBias = self.compute_dLoss_dBias(dLoss_dOutput)
        return self.dLoss_dInput

    def convolution(self, input_layer, filter, bias, strides):
        output = np.zeros(shape=(input_layer.shape[0],
                                 filter.shape[0],
                                 (input_layer.shape[2] - filter.shape[2]) // strides + 1,
                                 (input_layer.shape[3] - filter.shape[3]) // strides + 1)
                          )
        for bs in range(output.shape[0]):
            for ch in range(output.shape[1]):
                for i in range(output.shape[2]):
                    for j in range(output.shape[3]):
                        output[bs, ch, i, j] = np.sum(
                            np.multiply(
                                input_layer[
                                    bs,
                                    :,
                                    i * strides: i * strides + filter.shape[2],
                                    j * strides: j * strides + filter.shape[3],
                                ],
                                filter[ch, :, :, :]
                            ) + bias[ch]
                        )
        return output

    def update_filter_and_bias(self, learning_rate):
        self.filter -= learning_rate * self.dLoss_dFilter
        self.bias -= learning_rate * self.dLoss_dBias


class Max_Pool_Layer(Zero_Padding):

    def __init__(self, ksize, strides, padding):
        super(Max_Pool_Layer, self).__init__()
        self.ksize = ksize
        self.strides = strides
        self.padding = padding

    def max_pool(self, input_layer, ksize, strides):
        output = np.zeros(shape=(input_layer.shape[0],
                                 input_layer.shape[1],
                                 (input_layer.shape[2] - ksize[2]) // strides + 1,
                                 (input_layer.shape[3] - ksize[3]) // strides + 1)
                          )

        max_value_mask = np.zeros(shape=input_layer.shape,
                                  dtype=bool)

        for bs in range(output.shape[0]):
            for ch in range(output.shape[1]):
                for i in range(output.shape[2]):
                    for j in range(output.shape[3]):
                        pool = input_layer[
                                    bs,
                                    ch,
                                    i * strides: i * strides + ksize[2],
                                    j * strides: j * strides + ksize[3]
                                ]
                        output[bs, ch, i, j] = np.max(pool)
                        max_relative_pos = np.unravel_index(indices=np.argmax(pool), dims=pool.shape)
                        max_value_mask[
                            bs,
                            ch,
                            i * strides + max_relative_pos[0],
                            j * strides + max_relative_pos[1]
                        ] = 1

        return output, max_value_mask

    def max_pool_prime(self, input_layer_shape, dLoss_dOutput, strides, max_value_mask, ksize):
        dLoss_dInput = np.zeros(shape=input_layer_shape)
        for bs in range(dLoss_dOutput.shape[0]):
            for ch in range(dLoss_dOutput.shape[1]):
                for i in range(dLoss_dOutput.shape[2]):
                    for j in range(dLoss_dOutput.shape[3]):
                        max_value_pool = max_value_mask[
                                             bs,
                                             ch,
                                             i * strides: i * strides + ksize[2],
                                             j * strides: j * strides + ksize[3]
                                         ]
                        max_relative_pos = np.unravel_index(indices=np.argmax(max_value_pool),
                                                            dims=max_value_pool.shape)
                        dLoss_dInput[
                            bs,
                            ch,
                            i * strides + max_relative_pos[0],
                            j * strides + max_relative_pos[1]
                        ] = dLoss_dOutput[bs, ch, i, j]

        return dLoss_dInput

    def forward_propagation(self, input_layer):

        self.input_layer = input_layer

        if self.padding == 'SAME':
            self.input_layer_zp, self.input_layer_origin_mask = self.input_layer_zero_padding(self.input_layer,
                                                                                              self.ksize,
                                                                                              self.strides)
        input_layer = self.input_layer_zp if self.padding == 'SAME' else self.input_layer
        output, self.max_value_mask = self.max_pool(input_layer, self.ksize, self.strides)
        return output

    def backward_propagation(self, dLoss_dOutput):
        input_layer = self.input_layer_zp if self.padding == 'SAME' else self.input_layer
        dLoss_dInput = self.max_pool_prime(input_layer.shape, dLoss_dOutput, self.max_value_mask, self.strides,
                                           self.ksize)
        return dLoss_dInput


class Flatten_Layer:

    def __init__(self):
        pass

    def forward_propagation(self, input_layer):
        self.input_layer_shape = input_layer.shape
        output = np.reshape(input_layer, (input_layer.shape[0], -1))
        return output

    def backward_propagation(self, dLoss_dOutput):
        dLoss_dInput = np.reshape(dLoss_dOutput, self.input_layer_shape)
        return dLoss_dInput


class Relu_Layer:

    def __init__(self):
        pass

    def forward_propagation(self, input_layer):
        self.input_layer = input_layer
        return np.where(self.input_layer >= 0, self.input_layer, 0)

    def backward_propagation(self, dLoss_dOutput):
        dLoss_dInput = np.where(self.input_layer >= 0, dLoss_dOutput, 0)
        return dLoss_dInput


class Sigmoid_Layer:

    def __init__(self):
        pass

    def forward_propagation(self, input_layer):
        self.output = 1 / (1 + np.exp(-input_layer))
        return self.output

    def backward_propagation(self, dLoss_dOutput):
        dOutput_dInput = np.multiply(self.output, (1 - self.output))
        dLoss_dInput = np.multiply(dLoss_dOutput, dOutput_dInput)
        return dLoss_dInput


class Dropout_Layer:

    def __init__(self):
        pass

    def forward_propagation(self, input_layer, keep_rate):
        self.keep_rate = keep_rate
        self.mask = [np.random.uniform(size=input_layer.shape[1]) <= self.keep_rate]
        return np.where(self.mask, input_layer / self.keep_rate, 0)

    def backward_propagation(self, dLoss_dOutput):
        return np.where(self.mask, dLoss_dOutput / self.keep_rate, 0)


class Softmax_Layer:

    def __init__(self):
        pass

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

    def __init__(self):
        pass

    def forward_propagation(self, predict, one_hot_label):
        self.predict = predict
        self.one_hot_label = one_hot_label
        self.small_number = 1e-10

        self.output = np.mean(
                          np.sum(
                              - self.one_hot_label * np.log(self.predict + self.small_number)
                              - (1 - self.one_hot_label) * np.log((1 - self.predict + self.small_number)),
                          axis=1)
                       )
        return self.output

    def backward_propagation(self):

        dLoss_dPredict = (1 / self.one_hot_label.shape[0]) * \
                            ((- self.one_hot_label / (self.predict + self.small_number)) +
                             ((1 - self.one_hot_label) / (1 - self.predict + self.small_number)))
        return dLoss_dPredict













