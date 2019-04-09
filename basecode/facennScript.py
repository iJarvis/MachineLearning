'''
Comparing single layer MLP with deep MLP (using TensorFlow)
'''

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

# Do not change this
def initializeWeights(n_in,n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
                            
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""
    epsilon = sqrt(6) / sqrt(n_in + n_out + 1)
    W = (np.random.rand(n_out, n_in + 1)*2* epsilon) - epsilon
    return W



# Replace this with your sigmoid implementation
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-1.0 * z))
    
# Replace this with your nnObjFunction implementation
def nnObjFunction(params, *args):
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
    
# Replace this with your nnPredict implementation
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

# Do not change this
def preprocess():
    pickle_obj = pickle.load(file=open('face_all.pickle', 'rb'))
    features = pickle_obj['Features']
    labels = pickle_obj['Labels']
    train_x = features[0:21100] / 255
    valid_x = features[21100:23765] / 255
    test_x = features[23765:] / 255

    labels = labels[0]
    train_y = labels[0:21100]
    valid_y = labels[21100:23765]
    test_y = labels[23765:]
    return train_x, train_y, valid_x, valid_y, test_x, test_y

"""**************Neural Network Script Starts here********************************"""
train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()
#  Train Neural Network
# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]
# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 256
# set the number of nodes in output unit
n_class = 2

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden)
initial_w2 = initializeWeights(n_hidden, n_class)
# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()),0)
# set the regularization hyper-parameter
lambdaval = 10
args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

#Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example
opts = {'maxiter' :50}    # Preferred value.

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args,method='CG', options=opts)
params = nn_params.get('x')
#Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = params[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

#Test the computed parameters
predicted_label = nnPredict(w1,w2,train_data)
#find the accuracy on Training Dataset
print('\n Training set Accuracy:' + str(100*np.mean((predicted_label == train_label).astype(float))) + '%')
predicted_label = nnPredict(w1,w2,validation_data)
#find the accuracy on Validation Dataset
print('\n Validation set Accuracy:' + str(100*np.mean((predicted_label == validation_label).astype(float))) + '%')
predicted_label = nnPredict(w1,w2,test_data)
#find the accuracy on Validation Dataset
print('\n Test set Accuracy:' +  str(100*np.mean((predicted_label == test_label).astype(float))) + '%')