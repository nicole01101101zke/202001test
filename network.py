#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split



def init_parameters(layer_dims):
    np.random.seed(3)
    L = len(layer_dims)
    parameters = {}
    for l in range(1,L):
        #parameters["W" + str(l)] = np.random.randn(layer_dims[l],layer_dims[l-1])*0.1
        # parameters["W" + str(l)] = np.zeros((layer_dims[l], layer_dims[l - 1])) 
        #parameters["W" + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1])*np.sqrt(2/layer_dims[l-1]) 
        parameters["W" + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * np.sqrt(1 / layer_dims[l - 1]) 
        parameters["b" + str(l)] = np.zeros((layer_dims[l],1))
    return parameters

def linear_forward(x, w, b):
    #print(w)
    #print(x)
    #print(w.shape)
    #print(x.shape)
    #print(type(w[0][0]))
    #print(type(x[0][0]))
    z = np.dot(w,x) + b
    return z

def relu_forward(Z):
    A = np.maximum(0,Z)
    return A

#implement the activation function(ReLU and sigmoid)
def sigmoid(Z):
    Z = np.array(Z,dtype=np.float64)
    A = 1 / (1 + np.exp(-Z))
    return A

def forward_propagation(X, parameters):
    L = len(parameters) // 2
    A = X
    caches = []
    for l in range(1,L):
        W = parameters["W" + str(l)]
        b = parameters["b" + str(l)]
        z = linear_forward(A, W, b)
        caches.append((A, W, b, z))
        A = relu_forward(z)
    WL = parameters["W" + str(L)]
    bL = parameters["b" + str(L)]
    zL = linear_forward(A, WL, bL)
    caches.append((A, WL, bL, zL))
    AL = sigmoid(zL)
    return AL, caches

def compute_cost(AL,Y):
    m = Y.shape[1]
    cost = 1. / m * np.nansum(np.multiply(-np.log(AL), Y) + np.multiply(-np.log(1 - AL), 1 - Y))
    cost = np.squeeze(cost)
    return cost


def relu_backward(dA, Z):
    dout = np.multiply(dA, np.int64(Z > 0))
    return dout

def linear_backward(dZ, cache):
    A, W, b, z = cache
    dW = np.dot(dZ, A.T)
    db = np.sum(dZ, axis=1, keepdims=True)
    da = np.dot(W.T, dZ)
    return da, dW, db


def backward_propagation(AL, Y, caches):
    m = Y.shape[1]
    L = len(caches) - 1
    dz = 1. / m * (AL - Y)
    da, dWL, dbL = linear_backward(dz, caches[L])
    gradients = {"dW" + str(L + 1): dWL, "db" + str(L + 1): dbL}

    for l in reversed(range(0,L)): 
        A, W, b, z = caches[l]
        dout = relu_backward(da, z)
        da, dW, db = linear_backward(dout, caches[l])
        # print(dW.shape)
        gradients["dW" + str(l+1)] = dW
        gradients["db" + str(l+1)] = db
    return gradients

def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2
    for l in range(L):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l+1)]
    return parameters

def L_layer_model(X, Y, layer_dims, learning_rate, num_iterations):
    costs = []
    parameters = init_parameters(layer_dims)
    for i in range(0, num_iterations):
        #foward propagation
        AL,caches = forward_propagation(X, parameters)
        # calculate the cost
        cost = compute_cost(AL, Y)
        if i % 1000 == 0:
            print("Cost after iteration {}: {}".format(i, cost))
            costs.append(cost)
        #backward propagation
        grads = backward_propagation(AL, Y, caches)
        #update parameters
        parameters = update_parameters(parameters, grads, learning_rate)
    print('length of cost')
    print(len(costs))
    return parameters

#predict function
def predict(X_test,y_test,parameters):
    m = y_test.shape[1]
    Y_prediction = np.zeros((1, m))
    prob, caches = forward_propagation(X_test,parameters)
    for i in range(prob.shape[1]):
        if prob[0, i] > 0.5:
            Y_prediction[0, i] = 1
        else:
            Y_prediction[0, i] = 0
    accuracy = 1- np.mean(np.abs(Y_prediction - y_test))
    return accuracy

#DNN model
def DNN(X_train, y_train, X_test, y_test, layer_dims, learning_rate= 0.001, num_iterations=40000):
    parameters = L_layer_model(X_train, y_train, layer_dims, learning_rate, num_iterations)
    accuracy = predict(X_test,y_test,parameters)
    return accuracy

if __name__ == "__main__":
    data = pd.read_csv('D:\大二下\专业英语\project2\dataset_mushrooms.csv',index_col=False, header=0,names=['target','x1','x2','x3','x4','x5','x6','x7','x8','x9','x10','x11','x12','x13','x14','x15','x16','x17','x18','x19','x20','x21','x22'],engine='python')
    
    X_data = data[['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10','x11','x12','x13','x14','x15','x16','x17','x18','x19','x20','x21','x22']].values
    y_data = data['target'].values
    y_data = np.array(y_data)
    X_data[X_data=='?'] = 0.0
    X_data[X_data=='a'] = 1.0
    X_data[X_data=='b'] = 2.0
    X_data[X_data=='c'] = 3.0
    X_data[X_data=='d'] = 4.0
    X_data[X_data=='e'] = 5.0
    X_data[X_data=='f'] = 6.0
    X_data[X_data=='g'] = 7.0
    X_data[X_data=='h'] = 8.0
    X_data[X_data=='i'] = 9.0
    X_data[X_data=='j'] = 10.0
    X_data[X_data=='k'] = 11.0
    X_data[X_data=='l'] = 12.0
    X_data[X_data=='m'] = 13.0
    X_data[X_data=='n'] = 14.0
    X_data[X_data=='o'] = 15.0
    X_data[X_data=='p'] = 16.0
    X_data[X_data=='q'] = 17.0
    X_data[X_data=='r'] = 18.0
    X_data[X_data=='s'] = 19.0
    X_data[X_data=='t'] = 20.0
    X_data[X_data=='u'] = 21.0
    X_data[X_data=='v'] = 22.0
    X_data[X_data=='w'] = 23.0
    X_data[X_data=='x'] = 24.0
    X_data[X_data=='y'] = 25.0
    X_data[X_data=='z'] = 26.0
    
    y_data[y_data=='p'] = 0
    y_data[y_data=='e'] = 1
    
    print(X_data)
    print(y_data)
    print(isinstance(y_data,np.ndarray))
    X_data = X_data.astype(np.float64)
    X_train, X_test,y_train,y_test = train_test_split(X_data, y_data, train_size=0.8,random_state=28)
    X_train = X_train.T
    print(y_train)
    print(isinstance(y_train,np.ndarray))
    y_train = y_train.reshape(y_train.shape[0], -1).T
    X_test = X_test.T
    y_test = y_test.reshape(y_test.shape[0], -1).T
    print(X_train)
    print(type(y_train[0][0]))
    print(y_train)
    print(X_test)
    print(y_test)
    accuracy = DNN(X_train,y_train,X_test,y_test,[X_train.shape[0],10,5,1])
    print(accuracy)


# In[ ]:





# In[ ]:




