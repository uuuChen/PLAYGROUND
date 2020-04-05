import numpy as np
import os

class Model:

    def __init__(self, pars):
        self.pars = pars
        self.w = None

    def _sigmoid(self, Z):
        # add small number to avoid overflow
        Z = Z.astype(float)
        SMALL_NUMBER = 1e-10
        return 1/(1 + np.exp(-1 * Z + SMALL_NUMBER))

    def _LR_gradient_descent(self, w, X, y):
        SMALL_NUMBER = 1e-10
        LEARNING_RATE = 0.01
        TOTAL_STEP = 400000
        PRINT_STEP = 10000
        step = 0

        while True:
            y_hat = np.dot(X, w)
            sig_y_hat = self._sigmoid(y_hat)

            # loss_value = (-1 / m) * (ylogy + (1 - y)log(1 - y))
            loss_value = -1 * np.mean((np.multiply(y, np.log(sig_y_hat + SMALL_NUMBER)) + np.multiply((1 - y), np.log(
                1 - sig_y_hat + SMALL_NUMBER))))

            # w_gradient = (1 / m) * (X.T * (y_hat - y))
            w_gradient = np.dot(X.T, (sig_y_hat - y)) / X.shape[0]
            w = w - LEARNING_RATE * w_gradient
            step += 1
            if step % PRINT_STEP == 0:
                y_hat_prime = np.where(sig_y_hat >= 0.5, 1, 0)
                accuracy = np.mean(y == y_hat_prime)
                print('step : %s \t lose_value : %s \t accuracy : %s' % (step, loss_value, accuracy))
            if step == TOTAL_STEP:
                return w

    def train(self, X, y):
        X, y = X.astype(float), y.astype(float)
        XT_X_edge = X_col = X.shape[1]
        w_file_path = 'w.txt'

        # add an '1' column at the right end of the X matrix when using bias
        if self.pars['bias']:
            X = np.insert(X, X_col, 1.0, axis=1)
            XT_X_edge = X_col + 1

        # linear regression
        if self.pars['model_type'] == 'lr':
            if self.pars['regulation']:
                # w = ((X.T * X + λI) ^ -1) * X.T * y
                lambda_I = np.identity(XT_X_edge) * self.pars['reg_lambda']
                w = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X) + lambda_I), X.T), y)

            else:
                # w = ((X.T * X) ^ -1) * X.T * y
                w = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y)

        # bayesian linear regression
        elif self.pars['model_type'] == 'blr':
            # w = ((X.T * X + λI) ^ -1) * X.T * y
            lambda_I = np.identity(XT_X_edge) * self.pars['bay_lambda']
            w = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X) + lambda_I), X.T), y)

        # logistic regression
        # use gradient descent to find the optimal w, and store w in 'w.txt' at the first time
        elif self.pars['model_type'] == 'LR':
            if os.path.isfile(w_file_path):
                w = self._read_w_from_file(file_name=w_file_path)

            else:
                w0 = np.random.normal(loc=0, scale=100, size=(X_col))
                w = self._LR_gradient_descent(w0, X, y)
                self._write_w_to_file(file_name=w_file_path, w=w)

        # store w in this class
        self.w = w

    def _read_w_from_file(self, file_name):
        # print('\n---read w from file ' + file_name +'---\n')
        file = open(file_name, 'r')
        w = np.array(file.readline().split())
        return w

    def _write_w_to_file(self, file_name,  w):
        # print('---write w to %s---', file_name)
        file = open(file_name, 'a')
        file.write(str(w))
        file.close()

    def write_yhat_to_file(self, file_name, X):
        file = open(file_name, "w")
        y_hat = self.get_y_hat(X)
        y_hat_idx = np.arange(0, len(y_hat))
        START_ID = 1001
        id_idx = np.arange(START_ID, START_ID + len(y_hat))
        for y_hat_idx, id_idx in zip(y_hat_idx, id_idx):
            s = str(id_idx) + '\t' + str(np.round(y_hat[y_hat_idx], 1)) + '\n'
            file.write(s)
        file.close()

    def get_w(self):
        if self.w.all == None:
            print('you have to train model first !')
            return None
        else:
            return self.w.astype(float)

    def get_y_hat(self, X):
        if self.pars['bias']:
            X = np.insert(X, len(X[0]), 1, axis=1)
        y_hat = np.dot(X, self.get_w())
        if self.pars['model_type'] == 'LR':
            y_hat = self._sigmoid(y_hat)
        if self.pars['binary_label']:
            y_hat = np.where(y_hat >= self.pars['yhat_threshold'], 1, 0)
        return y_hat

    def get_RMSE(self, X, y):
        y_hat = self.get_y_hat(X)
        RMSE = np.sqrt(np.mean(np.power((y - y_hat), 2)))
        return RMSE

    def get_confusion_matrix(self, X, y):
        y_hat = self.get_y_hat(X)
        c_m = np.zeros([2, 2])
        for i in range(0, len(y_hat)):
            if y_hat[i] == y[i]:
                if y[i] == 0:
                    c_m[0][0] += 1
                else:
                    c_m[1][1] += 1
            else:
                if y[i] == 1:
                    c_m[0][1] += 1
                else:
                    c_m[1][0] += 1
        return c_m

    def set_binary_label(self, binary_label):
        self.pars['binary_label'] = binary_label

    def set_yhat_threshold(self, yhat_threshold):
        self.pars['yhat_threshold'] = yhat_threshold

    def console(self, RMSE):
        log = self.pars['log']
        if self.pars['yhat_threshold'] == None:
            print('RMSE of %s is %s' % (log, RMSE))
        else:
            print('RMSE of %s and threshold = %s is %s' % (log, self.pars['yhat_threshold'], RMSE))












