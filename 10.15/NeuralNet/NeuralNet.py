#####################################################################################################################
#   CS 6375.003 - Assignment 3, Neural Network Programming
#   This is a starter code in Python 3.6 for a 2-hidden-layer neural network.
#   You need to have numpy and pandas installed before running this code.
#   Below are the meaning of symbols:
#   train - training dataset - can be a link to a URL or a local file
#         - you can assume the last column will the label column
#   train - test dataset - can be a link to a URL or a local file
#         - you can assume the last column will the label column
#   h1 - number of neurons in the first hidden layer
#   h2 - number of neurons in the second hidden layer
#   X - vector of features for each instance
#   y - output for each instance
#   w01, delta01, X01 - weights, updates and outputs for connection from layer 0 (input) to layer 1 (first hidden)
#   w12, delata12, X12 - weights, updates and outputs for connection from layer 1 (first hidden) to layer 2 (second hidden)
#   w23, delta23, X23 - weights, updates and outputs for connection from layer 2 (second hidden) to layer 3 (output layer)
#
#   You need to complete all TODO marked sections
#   You are free to modify this code in any way you want, but need to mention it in the README file.
#
#####################################################################################################################


import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, scale, MinMaxScaler
from sklearn.model_selection import train_test_split

class NeuralNet:
    def __init__(self, train, header = None, h1 = 4, h2 = 2, split_ratio=0.3):
        np.random.seed(1)
        # train refers to the training dataset
        # test refers to the testing dataset
        # h1 and h2 represent the number of nodes in 1st and 2nd hidden layers

        raw_input = pd.read_csv(train)
        # TODO: Remember to implement the preprocess method
        train_dataset = self.preprocess(raw_input)
        ncols = len(train_dataset.columns)
        nrows = len(train_dataset.index)

        self.X = train_dataset.iloc[:, 0:(ncols -1)].values.reshape(nrows, ncols-1)
        self.y = train_dataset.iloc[:, (ncols-1)].values.reshape(nrows, 1)
        self.trainX, self.testX, self.trainY, self.testY = train_test_split(self.X, self.y, test_size=split_ratio)
        #
        # Find number of input and output layers from the dataset
        #
        input_layer_size = len(self.X[0])
        if not isinstance(self.y[0], np.ndarray):
            output_layer_size = 1
        else:
            output_layer_size = len(self.y[0])

        # assign random weights to matrices in network
        # number of weights connecting layers = (no. of nodes in previous layer) x (no. of nodes in following layer)
        self.w01 = 2 * np.random.random((input_layer_size, h1)) - 1
        self.X01 = self.trainX
        self.delta01 = np.zeros((input_layer_size, h1))
        self.w12 = 2 * np.random.random((h1, h2)) - 1
        self.X12 = np.zeros((len(self.X), h1))
        self.delta12 = np.zeros((h1, h2))
        self.w23 = 2 * np.random.random((h2, output_layer_size)) - 1
        self.X23 = np.zeros((len(self.X), h2))
        self.delta23 = np.zeros((h2, output_layer_size))
        self.deltaOut = np.zeros((output_layer_size, 1))
    #
    # TODO: I have coded the sigmoid activation function, you need to do the same for tanh and ReLu
    #

    def __activation(self, x, activation="sigmoid"):
        if activation == "sigmoid":
            self.__sigmoid(self, x)
        if activation == "tanh":
            self.__tanh(self, x)
        if activation == "relu":
            self.__relu(self, x)

    #
    # TODO: Define the function for tanh, ReLu and their derivatives
    #

    def __activation_derivative(self, x, activation="sigmoid"):
        if activation == "sigmoid":
            self.__sigmoid_derivative(self, x)
        if activation == "tanh":
            self.__tanh_derivative(self, x)
        if activation == "relu":
            self.__relu_derivative(self, x)

    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def __tanh(self, x):
        return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))

    def __relu(self, x):
        _x = np.zeros(x.shape)
        nonzero_idxs = np.nonzero(x>0)
        _x[nonzero_idxs] = x[nonzero_idxs]
        return _x

    # derivative of sigmoid function, indicates confidence about existing weight

    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    def __tanh_derivative(self, x):
        return 1 - x*x

    def __relu_derivative(self, x):
        _x = np.zeros(x.shape)
        nonzero_idxs = np.nonzero(x>0)
        _x[nonzero_idxs] = 1*len(nonzero_idxs)
        return _x
    #
    # TODO: Write code for pre-processing the dataset, which would include standardization, normalization,
    #   categorical to numerical, etc
    #

    def preprocess(self, X):

        _df = pd.DataFrame()

        for i in range(len(X.columns)):
            c = X.columns[i]

            col = X[c].values
            if type(X[c].values[0]) is not str:
                new_col = scale(col)
                if i == len(X.columns) - 1:
                    scaler = MinMaxScaler()
                    new_col = scaler.fit_transform(np.reshape(col, (len(col), 1)), (0, 1))
                    new_col = new_col.reshape((len(col)))
                _df[str(c)] = pd.Series(new_col)
            else:
                lb_enc = LabelEncoder()
                new_col = lb_enc.fit_transform(X[c].values)
                if i < len(X.columns) -1:
                    oh_enc = OneHotEncoder()
                    new_col = oh_enc.fit_transform(new_col.reshape((new_col.shape[0],1))).toarray()
                    for i in range(new_col.shape[1]):
                        _df[str(c)+'_'+str(i+1)] = pd.Series(new_col[:,i])
                else:
                    scaler = MinMaxScaler()
                    new_col = scaler.fit_transform(new_col.reshape((new_col.shape[0], 1)), (0,1))
                    _df[str(c)] = pd.Series(new_col.reshape((len(new_col))))

        return _df

    def acc_measure(self, pred, true):
        round_pred = np.round(pred)
        return 1.0 - np.nonzero(round_pred - true)[0].shape[0]/float(true.shape[0])

    def err_measure(self, pred, true):
        return 0.5 * np.power((pred - true), 2)

    # Below is the training function

    def train(self, activation="sigmoid", max_iterations = 1000, learning_rate = 0.05, elps=1e-6):
        self.activation = activation

        _err = 0
        for iteration in range(max_iterations):

            out = self.forward_pass(self.trainX)
            self.backward_pass(out, self.trainY)

            # print("out",out)
            # print("true",self.trainY)
            update_layer2 = learning_rate * self.X23.T.dot(self.deltaOut)
            update_layer1 = learning_rate * self.X12.T.dot(self.delta23)
            update_input = learning_rate * self.X01.T.dot(self.delta12)

            self.w23 += update_layer2
            self.w12 += update_layer1
            self.w01 += update_input

            error = self.err_measure(out, self.trainY)
            # print("- Iteration {} -- error:{}".format(iteration+1, np.sum(error)))

            if abs(np.sum(_err - error)) < elps:
                max_iterations = iteration + 1
                break
            _err = error

        print("### After " + str(max_iterations) + " iterations, the total error is " + str(np.sum(error)))
        print("### The final weight vectors are (starting from input to output layers):")
        print("- w01:\n{}".format(self.w01))
        print("- w12:\n{}".format(self.w12))
        print("- w23:\n{}".format(self.w23))

    def forward_pass(self, X):
        # pass our inputs through our neural network
        in1 = np.dot(X, self.w01 )
        if self.activation == "sigmoid":
            self.X12 = self.__sigmoid(in1)
            in2 = np.dot(self.X12, self.w12)
            self.X23 = self.__sigmoid(in2)
            in3 = np.dot(self.X23, self.w23)
            out = self.__sigmoid(in3)
        if self.activation == "tanh":
            self.X12 = self.__tanh(in1)
            print("out1:",self.X12)
            in2 = np.dot(self.X12, self.w12)
            self.X23 = self.__tanh(in2)
            print("out2:",self.X23)
            in3 = np.dot(self.X23, self.w23)
            out = self.__tanh(in3)
            print("out3",out)
        if self.activation == "relu":
            self.X12 = self.__relu(in1)
            in2 = np.dot(self.X12, self.w12)
            self.X23 = self.__relu(in2)
            in3 = np.dot(self.X23, self.w23)
            out = self.__relu(in3)
        return out



    def backward_pass(self, pred, true):
        # pass our inputs through our neural network
        self.compute_output_delta(pred, true)
        self.compute_hidden_layer2_delta()
        self.compute_hidden_layer1_delta()

    # TODO: Implement other activation functions

    def compute_output_delta(self, pred, true):
        if self.activation == "sigmoid":
            delta_output = (true - pred) * (self.__sigmoid_derivative(pred))
        if self.activation == "tanh":
            delta_output = (true- pred) * (self.__tanh_derivative(pred))
        if self.activation == "relu":
            delta_output = (true - pred) * (self.__relu_derivative(pred))

        self.deltaOut = delta_output

    # TODO: Implement other activation functions

    def compute_hidden_layer2_delta(self):
        if self.activation == "sigmoid":
            delta_hidden_layer2 = (self.deltaOut.dot(self.w23.T)) * (self.__sigmoid_derivative(self.X23))
        if self.activation == "tanh":
            delta_hidden_layer2 = (self.deltaOut.dot(self.w23.T)) * (self.__tanh_derivative(self.X23))
        if self.activation == "relu":
            delta_hidden_layer2 = (self.deltaOut.dot(self.w23.T)) * (self.__relu_derivative(self.X23))

        self.delta23 = delta_hidden_layer2

    # TODO: Implement other activation functions

    def compute_hidden_layer1_delta(self):
        if self.activation == "sigmoid":
            delta_hidden_layer1 = (self.delta23.dot(self.w12.T)) * (self.__sigmoid_derivative(self.X12))
        if self.activation == "tanh":
            delta_hidden_layer1 = (self.delta23.dot(self.w12.T)) * (self.__tanh_derivative(self.X12))
        if self.activation == "relu":
            delta_hidden_layer1 = (self.delta23.dot(self.w12.T)) * (self.__relu_derivative(self.X12))

        self.delta12 = delta_hidden_layer1

    # TODO: Implement other activation functions

    def compute_input_layer_delta(self):
        if self.activation == "sigmoid":
            delta_input_layer = np.multiply(self.__sigmoid_derivative(self.X01), self.delta01.dot(self.w01.T))
        if self.activation == "tanh":
            delta_input_layer = np.multiply(self.__tanh_derivative(self.X01), self.delta01.dot(self.w01.T))
        if self.activation == "relu":
            delta_input_layer = np.multiply(self.__relu_derivative(self.X01), self.delta01.dot(self.w01.T))

        self.delta01 = delta_input_layer

    # TODO: Implement the predict function for applying the trained model on the  test dataset.
    # You can assume that the test dataset has the same format as the training dataset
    # You have to output the test error from this function

    def predict(self, test=None, is_csv=False, header = None):
        if is_csv:
            df = pd.read_csv(test, header=header)
            input_data = self.preprocess(df)
            ncols = len(input_data.columns)
            nrows = len(input_data.index)
            X = input_data.iloc[:, 0:(ncols -1)].values.reshape(nrows, ncols-1)
            y = input_data.iloc[:, (ncols-1)].values.reshape(nrows, 1)
            pred = self.forward_pass(X)
        else:
            pred = self.forward_pass(self.testX)
            y = self.testY

        acc = self.acc_measure(pred, y)
        err = self.err_measure(pred, y)
        print("### Error on Test set is:{}\n### Accuracy on Test set is:{}".format(np.sum(err), acc))

        return pred


from config import *

if __name__ == "__main__":
    neural_network = NeuralNet(train_path, h1=4, h2=2, split_ratio=0)
    neural_network.train(activation="sigmoid", max_iterations=100000, learning_rate=0.1, elps=1e-12)
    # pred = neural_network.predict()