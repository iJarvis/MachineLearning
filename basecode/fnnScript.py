import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
from sklearn.preprocessing import OneHotEncoder


'''
Comparing single layer MLP with deep MLP (using TensorFlow)
'''

import numpy as np
import pickle

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
    epsilon = np.sqrt(6) / np.sqrt(n_in + n_out + 1)
    W = (np.random.rand(n_out, n_in + 1)*2* epsilon) - epsilon
    return W



# Replace this with your sigmoid implementation
def sigmoid(z):
    sigm = 1. / (1. + np.exp(-z))

    return  sigm# your code here


    
# Replace this with your nnObjFunction implementation
def nnObjFunction(params, *args):
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
    
# Replace this with your nnPredict implementation
def nnPredict(w1,w2,data):
    labels = np.array([])
    
    for instance in data:
        hidden_layer_ip = sigmoid(np.sum(w1*(np.hstack((instance, np.ones(1)))), axis = 1))
        output_layer = sigmoid(np.sum(w2*(np.hstack((hidden_layer_ip, np.ones(1)))), axis = 1))
        pred_val = np.argmax(output_layer)
        labels = np.append(labels, np.array([pred_val]))
    
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

import time
times = {} #type:dict
acc_train = {} #type:dict
acc_test = {} #type:dict
acc_validate = {} #type:dict
lambda_vals =[x for x in range(0,61,5)]
hidden_node_list = [4, 8, 12, 16, 20]

for hidden_node_num in hidden_node_list:
    
    args = (n_input, hidden_node_num, n_class, train_data, train_label, lambdaval)

    #Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example
    opts = {'maxiter' :50}    # Preferred value.

    start = time.time()
    nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args,method='CG', options=opts)
    end = time.time()
    times[hidden_node_num] = end-start
    params = nn_params.get('x')
    #Reshape nnParams from 1D vector into w1 and w2 matrices
    w1 = params[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

    #Test the computed parameters
    predicted_label = nnPredict(w1,w2,train_data)
    #find the accuracy on Training Dataset
    print('\n Training set Accuracy for '+str(hidden_node_num) + " hidden nodes: " + str(100*np.mean((predicted_label == train_label).astype(float))) + '%')
    predicted_label = nnPredict(w1,w2,validation_data)
    #find the accuracy on Validation Dataset
    print('\n Validation set Accuracy for '+str(hidden_node_num) + " hidden nodes : " + str(100*np.mean((predicted_label == validation_label).astype(float))) + '%')
    predicted_label = nnPredict(w1,w2,test_data)
    #find the accuracy on Validation Dataset
    print('\n Test set Accuracy for '+str(hidden_node_num) + " hidden nodes : " + str(100*np.mean((predicted_label == test_label).astype(float))) + '%')


for lambda_val in lambda_vals:
    
    args = (n_input, n_hidden, n_class, train_data, train_label, lambda_val)

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
    print('\n Training set Accuracy for '+str(lambda_val) + " lambda value : " + str(100*np.mean((predicted_label == train_label).astype(float))) + '%')
    acc_train[lambda_val] = 100*np.mean((predicted_label == train_label).astype(float)))
    predicted_label = nnPredict(w1,w2,validation_data)
    #find the accuracy on Validation Dataset
    print('\n Validation set Accuracy for '+str(lambda_val) + " lambda value : " + str(100*np.mean((predicted_label == validation_label).astype(float))) + '%')
    acc_validate[lambda_val] = 100*np.mean((predicted_label == validation_label).astype(float)))
    predicted_label = nnPredict(w1,w2,test_data)
    #find the accuracy on Validation Dataset
    acc_test[lambda_val] = 100*np.mean((predicted_label == test_label).astype(float)))
    print('\n Test set Accuracy for '+str(lambda_val) + " lambda value : " + str(100*np.mean((predicted_label == test_label).astype(float))) + '%')