# the main frame is like the benchmark_full from util.py
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from util import get_transformed_digit, forward, error_rate, cost, deri_w, deri_b, y2indicator
from sklearn.utils import shuffle

def sgd_stochstic():
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

    #stochastic:
    for i in range(500):
        X_shuffle,Y_train_shuffle = shuffle(X_train,yindi_train)
        for ii in range(min(N,500)):
            x_tem = X_shuffle[ii].reshape(1,D)
            y_tem = Y_train_shuffle[ii].reshape(1,10)
            y_fit = forward(x = x_tem,w=W,b=b)
            
            W += eta*(deri_w(t_matrix = y_tem, y_matrix = y_fit,x = x_tem)-penalty*W)
            b += eta*(deri_b(t_matrix = y_tem, y_matrix = y_fit)-penalty*b)

            p_y_test = forward(x = X_test,w=W,b=b)
            print(p_y_test)
            cost_test_tem = cost(y_matrix = p_y_test,t_matrix = yindi_test)
            cost_test.append(cost_test_tem)

            if ii % 100 == 0:
                error_tem = error_rate(y_matrix = p_y_test, target = Y_test)
                print("the error rate in "+str(ii)+" iteration is :"+str(error_tem))
    
    p_y_final = forward(x = X_test,w=W,b=b)
    error_final = error_rate(y_matrix = p_y_final, target = Y_test)
    print("the final error rate is "+str(error_final))

# sgd_stochstic()

def sgd_batch():
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

    batch_size = 500
    batch_num = N // batch_size

    #batch
    for i in range(500):
        X_shuffle,Y_train_shuffle = shuffle(X_train,yindi_train)
        for ii in range(int(batch_num)):
            # x_tem = X_shuffle[ii].reshape(1,D)
            # y_tem = Y_train_shuffle[ii].reshape(1,10)

            x_tem = X_shuffle[int(i*batch_size):int((i+1)*batch_size)]
            y_tem = Y_train_shuffle[int(i*batch_size):int((i+1)*batch_size)]

            y_fit = forward(x = x_tem,w=W,b=b)
            
            W += eta*(deri_w(t_matrix = y_tem, y_matrix = y_fit,x = x_tem)-penalty*W)
            b += eta*(deri_b(t_matrix = y_tem, y_matrix = y_fit)-penalty*b)

            p_y_test = forward(x = X_test,w=W,b=b)
            cost_test_tem = cost(y_matrix = p_y_test,t_matrix = yindi_test)
            cost_test.append(cost_test_tem)

            if ii % 100 == 0:
                error_tem = error_rate(y_matrix = p_y_test, target = Y_test)
                print("the error rate in "+str(ii)+" iteration is :"+str(error_tem))
    
    p_y_final = forward(x = X_test,w=W,b=b)
    error_final = error_rate(y_matrix = p_y_final, target = Y_test)
    print("the final error rate is "+str(error_final))


sgd_batch()



    # for i in range(10000):
    #     #train
    #     y_ma = forward(x=X_train,w=W,b=b)
    #     cost_tem = cost(y_matrix = y_ma,t_matrix = yindi_train)
    #     cost_train.append(cost_tem)
        
    #     #test
    #     y_test_ma = forward(x= X_test,w=W,b=b)
    #     cost_test_tem = cost(y_matrix = y_test_ma, t_matrix = yindi_test)
    #     cost_test.append(cost_test_tem)
        
    #     #error
    #     error_tem = error_rate(y_matrix = y_test_ma ,target = Y_test)
    #     error_test.append(error_tem)
        
    #     W += eta*(deri_w(t_matrix = yindi_train,y_matrix = y_ma,x = X_train) - penalty * W)
    #     b += eta*(deri_b(t_matrix = yindi_train,y_matrix = y_ma) - penalty * b)
    #     if i % 100 == 0:
    #         print("the error rate in "+str(i)+" iteration is : "+str(error_tem))
    
    # #final
    # y_final = forward(x = X_test, w=W,b=b)
    # error_final = error_rate(y_matrix = y_final,target = Y_test)
    # print("the final error rate is "+str(error_final))