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
from sklearn.preprocessing import scale, LabelEncoder, OneHotEncoder, MinMaxScaler

class NeuralNet:
    def __init__(self, train, activation="sigmoid", header = True, h1 = 16, h2 = 8):
        np.random.seed(1)
        # train refers to the training dataset
        # test refers to the testing dataset
        # h1 and h2 represent the number of nodes in 1st and 2nd hidden layers

        self.activation = activation
        if header:
            raw_input = pd.read_csv(train)
        else:
            raw_input = pd.read_csv(train, header=None)
        # TODO: Remember to implement the preprocess method
        train_dataset = self.preprocess(raw_input)
        print(train_dataset.values.shape)
        ncols = len(train_dataset.columns)
        nrows = len(train_dataset.index)
        self.X = train_dataset.iloc[:, 0:(ncols -1)].values.reshape(nrows, ncols-1)
        self.y = train_dataset.iloc[:, (ncols-1)].values.reshape(nrows, 1)
        if self.activation == "tanh":
            self.y = (self.y - 0.5) * 2
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
        self.X01 = self.X
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

    def __activation(self, x):
        if self.activation == "sigmoid":
            self.__sigmoid(self, x)
        if self.activation == "tanh":
            self.__tanh(self, x)
        if self.activation == "relu":
            self.__relu(self, x)

    #
    # TODO: Define the function for tanh, ReLu and their derivatives
    #

    def __activation_derivative(self, x):
        if self.activation == "sigmoid":
            self.__sigmoid_derivative(self, x)
        if self.activation == "tanh":
            self.__tanh_derivative(self, x)
        if self.activation == "relu":
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
                # if i < len(X.columns) -1:
                #     oh_enc = OneHotEncoder()
                #     new_col = oh_enc.fit_transform(new_col.reshape((new_col.shape[0],1))).toarray()
                #     for i in range(new_col.shape[1]):
                #         _df[str(c)+'_'+str(i+1)] = pd.Series(new_col[:,i])
                if i == len(X.columns) - 1:
                    scaler = MinMaxScaler()
                    new_col = scaler.fit_transform(new_col.reshape((new_col.shape[0], 1)), (0, 1))
                    _df[str(c)] = pd.Series(new_col.reshape((len(new_col))))
                else:
                    _df[str(c)] = pd.Series(new_col)

        return _df

    # Below is the training function

    def train(self, max_iterations = 1000, learning_rate = 0.02):

        for iteration in range(max_iterations):
            out = self.forward_pass()
            error = 0.5 * np.power((out - self.y), 2)
            self.backward_pass(out)
            update_layer2 = learning_rate * self.X23.T.dot(self.deltaOut)
            update_layer1 = learning_rate * self.X12.T.dot(self.delta23)
            update_input = learning_rate * self.X01.T.dot(self.delta12)
            # print(update_layer2)
            # print(update_layer1)
            # print(update_input)

            self.w23 += update_layer2
            self.w12 += update_layer1
            self.w01 += update_input
            print("err:",np.average(error))

        print("After " + str(max_iterations) + " iterations, the total error is " + str(np.sum(error)))
        print("The final weight vectors are (starting from input to output layers)")
        print(self.w01)
        print(self.w12)
        print(self.w23)

    def forward_pass(self):
        # pass our inputs through our neural network
        in1 = np.dot(self.X, self.w01 )
        if self.activation == "sigmoid":
            self.X12 = self.__sigmoid(in1)
            in2 = np.dot(self.X12, self.w12)
            self.X23 = self.__sigmoid(in2)
            in3 = np.dot(self.X23, self.w23)
            out = self.__sigmoid(in3)
        if self.activation == "tanh":
            self.X12 = self.__tanh(in1)
            in2 = np.dot(self.X12, self.w12)
            self.X23 = self.__tanh(in2)
            in3 = np.dot(self.X23, self.w23)
            out = self.__tanh(in3)
        if self.activation == "relu":
            self.X12 = self.__relu(in1)
            in2 = np.dot(self.X12, self.w12)
            self.X23 = self.__relu(in2)
            in3 = np.dot(self.X23, self.w23)
            out = self.__relu(in3)
        return out



    def backward_pass(self, out):
        # pass our inputs through our neural network
        self.compute_output_delta(out)
        self.compute_hidden_layer2_delta()
        self.compute_hidden_layer1_delta()

    # TODO: Implement other activation functions

    def compute_output_delta(self, out):
        if self.activation == "sigmoid":
            delta_output = (self.y - out) * (self.__sigmoid_derivative(out))
        if self.activation == "tanh":
            delta_output = (self.y - out) * (self.__tanh_derivative(out))
        if self.activation == "relu":
            delta_output = (self.y - out) * (self.__relu_derivative(out))

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

    def predict(self, test, header = True):
        df = pd.read_csv(test)
        _df = self.preprocess(df)
        ncols = len(_df.columns)
        nrows = len(_df.index)
        self.X = _df.iloc[:, 0:(ncols -1)].values.reshape(nrows, ncols-1)
        y = _df.iloc[:, (ncols-1)].values.reshape(nrows, 1)
        pred = self.forward_pass()
        print("pred:\n",pred)
        print("true:\n",y)
        return 0.5 * np.power((y - pred),2)

import config
if __name__ == "__main__":
    neural_network = NeuralNet(config.adultData_path, activation="sigmoid", header=False, h1=10, h2=10)
    neural_network.train(max_iterations=1000, learning_rate=0.2)
    # testError = neural_network.predict("test.csv")
    # neural_network.predict("test.csv")

