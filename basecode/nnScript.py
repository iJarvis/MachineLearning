import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
from time import strftime, localtime
import sys
import pickle

def getOutputs(w1, w2, data):
  """
  Caluculates the hidden and output layer output values, given set of weights and input
  """
  data = data.T
  bias = np.ones((1, data.shape[1]), dtype = np.int)
  data = np.concatenate((data, bias), axis = 0)

  hidden = sigmoid(np.dot(w1, data))
  hidden_bias = np.ones((1, hidden.shape[1]), dtype = np.int)
  hidden = np.concatenate((hidden, hidden_bias), axis = 0)

  output = sigmoid(np.dot(w2, hidden))
  
  return (data, hidden, output)

def initializeWeights(n_in, n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
       
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""

    epsilon = sqrt(6) / sqrt(n_in + n_out + 1)
    W = (np.random.rand(n_out, n_in + 1) * 2 * epsilon) - epsilon
    return W


def sigmoid(z):
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""

    return 1.0 / (1.0 + np.exp(-1.0 * z))


def preprocess():
    """ Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set

     Some suggestions for preprocessing step:
     - feature selection"""

    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary

    # Split the training sets into two sets of 50000 randomly sampled training examples and 10000 validation examples. 
    # Your code here.
    # Feature selection
    # Your code here.
    training_data = np.zeros((0,784))
    test_data = np.zeros((0,784))
    training_labels = np.zeros((0,))
    test_labels = np.zeros(0,)

    for i in range(10):
      training_data = np.vstack ((training_data, mat['train' + str(i)]))
      test_data = np.vstack ((test_data, mat['test' + str(i)]))
      training_labels = np.hstack((training_labels, i * np.ones(mat['train' + str(i)].shape[0])))
      test_labels = np.hstack((test_labels, i * np.ones(mat['test' + str(i)].shape[0])))
    
    training_data = training_data.astype(np.float)
    test_data = test_data.astype(np.float)
    training_data = training_data/255
    test_data = test_data/255
    
    train_indices = np.random.permutation(60000)

    train_data = training_data[train_indices[0:50000],:]
    validation_data = training_data[train_indices[50000:],:]
    train_labels = training_labels[train_indices[0:50000]]
    validation_labels = training_labels[train_indices[50000:]]

    useless_columns = []
    for i in range(784):
      if np.unique(train_data[:,i]).size == 1:
        useless_columns.append(i)
    
    train_data = np.delete(train_data, useless_columns, axis = 1)
    validation_data = np.delete(validation_data, useless_columns, axis= 1)
    test_data = np.delete(test_data, useless_columns, axis = 1)

    print("Preprocessing done")
    return train_data, train_labels, validation_data, validation_labels, test_data, test_labels


def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log 
    %   likelihood error function with regularization) given the parameters 
    %   of Neural Networks, thetraining data, their corresponding training 
    %   labels and lambda - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.
       
    % Output: 
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input 
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden 
    %     layer to unit i in output layer."""

    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args

    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0

    train_size = training_data.shape[0]

    error = 0.0

    reference = np.zeros((train_size, n_class), dtype = np.int)

    for i in range(train_size):
      reference[i][int(training_label[i])] = 1.0
    
    reference = reference.transpose()
    data, hidden_output, output = getOutputs(w1, w2, training_data)

    error_func = reference*np.log(output) + (1 - reference)*np.log(1 - output) 
    error = -1 * (np.sum(error_func[:]) / train_size)

    delta_output = output - reference
    w2_grad = np.dot(delta_output, hidden_output.T)

    hidden_delta = np.dot(w2.T, delta_output) * (hidden_output * (1 - hidden_output))
    w1_grad = np.dot(hidden_delta, data.T)
    w1_grad = w1_grad[:-1,:]

    sum_squares_weight = np.sum(np.sum(w1**2)) + np.sum(np.sum(w2**2))
    error += (lambdaval * sum_squares_weight) / (2.0*train_size)
    
    w1_grad = (w1_grad + lambdaval * w1) / train_size
    w2_grad = (w2_grad + lambdaval * w2) / train_size
    
    obj_grad = np.concatenate((w1_grad.flatten(), w2_grad.flatten()),0)
    obj_val = error
  
    return (obj_val,obj_grad)


def nnPredict(w1, w2, data):
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % w2: matrix of weights of cokl///   nnections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % data: matrix of data. Each row of this matrix represents the feature 
    %       vector of a particular image
       
    % Output: 
    % label: a column vector of predicted labels"""

    labels = np.array([])
    # Your code here
    # for instance in data:
    #   (hidden, output) = getOutputs(w1, w2, instance)
    #   prediction = np.argmax(output)
    #   labels = np.append(labels, np.array([prediction]))
    # return labels
    (data, hidden, output) = getOutputs(w1, w2, data)
    labels = np.argmax(output, axis = 0)

    return labels


"""**************Neural Network Script Starts here********************************"""

train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()
#  Train Neural Network

# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]

# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 50

# set the number of nodes in output unit
n_class = 10

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden)
initial_w2 = initializeWeights(n_hidden, n_class)

# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()), 0)

# set the regularization hyper-parameter
lambdaval = 0

args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

# Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

opts = {'maxiter': 50}  # Preferred value.

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)

# # In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
# # and nnObjGradient. Check documentation for this function before you proceed.
# # nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)


# Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
 
# Test the computed parameters

predicted_label = nnPredict(w1, w2, train_data)

# find the accuracy on Training Dataset

print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

predicted_label = nnPredict(w1, w2, validation_data)

# find the accuracy on Validation Dataset

print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

predicted_label = nnPredict(w1, w2, test_data)

# find the accuracy on Validation Dataset

print('\n Test set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')