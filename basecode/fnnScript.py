import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
from sklearn.preprocessing import OneHotEncoder


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
    
    sigm = 1. / (1. + np.exp(-z))

    return  sigm# your code here


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
    
    # Get test data
    dim1 = 0
    dim2 = 785

    for i in range(10):
        tst_var = "test"+str(i)
        tst_dat = mat.get(tst_var, "")
        dim1 = dim1 + tst_dat.shape[0]

    test_data = np.zeros((dim1, dim2))

    for i in range(10):
        test_var = "test"+str(i)
        test_dat = mat.get(test_var, "")
        target_test_dat = np.full((mat.get(test_var, "").shape[0], 1), i)
        test_dat = np.hstack((test_dat, target_test_dat))
        if i == 0:
            test_data = test_dat
        else:
            test_data = np.vstack((test_data, test_dat))

    # Get train data

    dim1 = 0
    dim2 = 785

    for i in range(10):
        tr_var = "train"+str(i)
        tr_dat = mat.get(tr_var, "")
        dim1 = dim1 + tr_dat.shape[0]

    final_dat = np.zeros((dim1, dim2))

    for i in range(10):
        train_var = "train"+str(i)
        train_dat = mat.get(train_var, "")
        target_dat = np.full((mat.get(train_var, "").shape[0], 1), i)
        train_dat = np.hstack((train_dat, target_dat))
        if i == 0:
            final_dat = train_dat
        else:
            final_dat = np.vstack((final_dat, train_dat))

    feature_restored_col = []
    for i in range(final_dat.shape[1]):
        if np.unique(final_dat[:,i]).size != 1:
            feature_restored_col.append(i)

    final_dat = final_dat[:,feature_restored_col]

    valid_dat_idx = np.random.choice(final_dat.shape[0], size=(int(final_dat.shape[0]/6)), replace=False)

    train_dat_idx = np.setdiff1d(np.arange(1,final_dat.shape[0]), valid_dat_idx)


    train_data = final_dat[train_dat_idx, :]
    validation_data = final_dat[valid_dat_idx, :]
    train_label = train_data[:, (train_data.shape[1] - 1)]
    validation_label = validation_data[:, (validation_data.shape[1] - 1)]
    test_data = test_data[:,feature_restored_col]
    test_label = test_data[:, (test_data.shape[1] - 1)]

    print('preprocess done')

    return train_data, train_label, validation_data, validation_label, test_data, test_label


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

    # Your code here
    train_size = training_data.shape[0]
    grad_w1 = 0.0
    grad_w2 = 0.0
    err = 0.0
    
    for i in range(train_size):
        instance = training_data[i]
        
        # Get the output for that instance
        hidden = sigmoid(np.sum(w1*(np.hstack((instance, np.ones(1)))), axis = 1))
        output = sigmoid(np.sum(w2*(np.hstack((hidden, np.ones(1)))), axis = 1))
        
        # One hot encoding of train labels
        X = training_label.reshape(-1,1)
        Y_l = np.zeros(n_class)
        Y_l[int(training_label[i])] = 1.0
        
        #print(output)
        
        err+=  np.sum((Y_l * np.log(output)) + ((1 - Y_l) * np.log(1 - output)))

        delta_l = output - Y_l
        
        grad_w2+= (delta_l.reshape((n_class, 1))) * np.hstack((hidden, np.ones((1))))
        
        summ_fact = np.dot(delta_l, w2)
        
        summ_fact = summ_fact[:-1]
        
        delta_j = (1 - hidden) * hidden * summ_fact
        
        grad_w1+= (delta_j.reshape((n_hidden, 1))) * np.hstack((instance, np.ones(1)))
        
    err = -err
    err = (err/train_size)
    
    sum_sq_wt = np.sum(np.sum(np.power(w2, 2))) + np.sum(np.sum(np.power(w1, 2)))
    err+= (lambdaval * sum_sq_wt)/(2.0 * train_size)
    
    grad_w1 = (1/train_size)*(grad_w1 + (lambdaval * w1))
    grad_w2 = (1/train_size)*(grad_w2 + (lambdaval * w2))
    
    obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    obj_val = err
    
    # Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    # you would use code similar to the one below to create a flat array
    # obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    #obj_grad = np.array([])

    return (obj_val, obj_grad)

def nnPredict(w1, w2, data):
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % data: matrix of data. Each row of this matrix represents the feature 
    %       vector of a particular image
       
    % Output: 
    % label: a column vector of predicted labels"""

    labels = np.array([])
    
    for instance in data:
        hidden_layer_ip = sigmoid(np.sum(w1*(np.hstack((instance, np.ones(1)))), axis = 1))
        output_layer = sigmoid(np.sum(w2*(np.hstack((hidden_layer_ip, np.ones(1)))), axis = 1))
        pred_val = np.argmax(output_layer)
        labels = np.append(labels, np.array([pred_val]))
    
    return labels


"""*****Neural Network Script Starts here***********"""

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

# In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
# and nnObjGradient. Check documentation for this function before you proceed.
# nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)


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