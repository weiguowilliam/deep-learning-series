#this is the file that contains the common functions used in ann series lesson.
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

def get_data():
    """
    get random samples with 3 classes, N-class = 500
    """
    N_class = 500
    K = 3

    X1 = np.random.rand(N_class,2) + np.array([2,2])
    X2 = np.random.rand(N_class, 2) + np.array([-2,0])
    X3 = np.random.rand(N_class, 2) +np.array([0,4])
    X = np.vstack((X1,X2,X3))
    
    Y = np.array([0]*N_class+[1]*N_class+[2]*N_class)

    return X,Y

def get_spiral():
    radius = np.linspace(1,10,100)
    theta = np.zeros((6,100))
    for i in range(6):
        start_angle = np.pi/3 * i
        end_angle = start_angle + np.pi/2
        tem = np.linspace(start_angle,end_angle,100)
        theta[i] = tem
    
    x1 = np.zeros((6,100))
    x2 = np.zeros((6,100))
    
    for i in range(6):
        x1[i] = radius * np.cos(theta[i])
        x2[i] = radius * np.sin(theta[i])
    
    X = np.zeros((600,2))
    X[:,0] = x1.flatten()
    X[:,1] = x2.flatten()
    
    X +=  np.random.rand(600,2)/5
    Y = np.array([0]*100+[1]*100+[0]*100+[1]*100+[0]*100+[1]*100)
    
    return X, Y

def get_transformed_digit():
    #import
    dt = pd.read_csv('digit.csv').values.astype(np.float32)
    np.random.shuffle(dt)
    X = dt[:,1:]
    Y = dt[:,0]
    
    #pca
    pca = PCA(n_components=100)
    Z = pca.fit_transform(X)
    exp_sum = np.sum(pca.explained_variance_ratio_)
    
    #normalize
    z_mean = np.mean(Z,axis=0)
    z_std = np.std(Z, axis = 0)
    Z_normalized = (Z-z_mean)/z_std
    
    #split
    X_train = Z_normalized[:-300]
    Y_train = Y[:-300]
    X_test = Z_normalized[-300:]
    Y_test = Y[-300:]
    
    #return
    return X_train, Y_train, X_test, Y_test

def forward(x,w,b):
    a = x.dot(w) + b
    a_exp = np.exp(a)
    return a_exp/np.sum(a_exp,axis=1,keepdims=True)

def predict(y_matrix):
    """
    transform the fitted y_matrix to indicator vector
    """
    return np.argmax(y_matrix, axis=1)

def error_rate(y_matrix,target):
    y_vector = np.argmax(y_matrix,axis = 1)
    return np.mean(y_vector != target)

def cost(y_matrix,t_matrix):
    """
    here t is n*k matrix, not vector
    """
    tot = t_matrix*np.log(y_matrix)
    return (-1)*np.sum(tot)

def deri_w(t_matrix,y_matrix,x):
    return x.T.dot(t_matrix - y_matrix)

def deri_b(t_matrix,y_matrix):
    return np.sum(t_matrix - y_matrix, axis = 0)

def y2indicator(y_vector):
    """
    transform the vector y to matrix y
    """
    l = len(y_vector)
    indi = np.zeros((l,10))
    for i in range(l):
        indi[i,int(y_vector[i])] = 1
    return indi

def benchmark_full():
    """
    use util functions to run the logistic classification with bp
    """
    
    X_train, Y_train, X_test, Y_test = get_transformed_digit()
    
    N,D = X_train.shape
    yindi_train = y2indicator(Y_train)
    yindi_test = y2indicator(Y_test)
    
    M = yindi_test.shape[1]
    
    W = np.random.rand(D,M)
    b = np.random.rand(M)
    
    cost_train = []
    cost_test = []
    error_test = []
    
    eta = 1e-4
    penalty = 1e-2
    for i in range(10000):
        #train
        y_ma = forward(x=X_train,w=W,b=b)
        cost_tem = cost(y_matrix = y_ma,t_matrix = yindi_train)
        cost_train.append(cost_tem)
        
        #test
        y_test_ma = forward(x= X_test,w=W,b=b)
        cost_test_tem = cost(y_matrix = y_test_ma, t_matrix = yindi_test)
        cost_test.append(cost_test_tem)
        
        #error
        error_tem = error_rate(y_matrix = y_test_ma ,target = Y_test)
        error_test.append(error_tem)
        
        W += eta*(deri_w(t_matrix = yindi_train,y_matrix = y_ma,x = X_train) - penalty * W)
        b += eta*(deri_b(t_matrix = yindi_train,y_matrix = y_ma) - penalty * b)
        if i % 100 == 0:
            print("the error rate in "+str(i)+" iteration is : "+str(error_tem))
    
    #final
    y_final = forward(x = X_test, w=W,b=b)
    error_final = error_rate(y_matrix = y_final,target = Y_test)
    print("the final error rate is "+str(error_final))