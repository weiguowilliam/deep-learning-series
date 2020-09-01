import numpy as np
from sklearn.utils import shuffle
from util import get_transformed_digit,  error_rate, cost,y2indicator
from mlp import forward, deri_w1, deri_w2, deri_b1, deri_b2
import time

def rmsprop():
    """
    revise from benchmark_batch.py
    """
    
    X_train, Y_train, X_test, Y_test = get_transformed_digit()
    
    N,D = X_train.shape
    yindi_train = y2indicator(Y_train)
    yindi_test = y2indicator(Y_test)
    
    M = 300
    K = 10
    
    # W = np.random.rand(D,M)
    # b = np.random.rand(M)
    W1 = np.random.rand(D,M)/np.sqrt(D)
    b1 = np.zeros(M)
    W2 = np.random.rand(M,K)/np.sqrt(M)
    b2 = np.zeros(K)

    cost_test = []
    error_test = []
    
    eta = 0.00004
    penalty = 0.001

    batch_size = 500
    batch_num = N // batch_size

    # two new variable
    decay = 0.9
    eps = 1e-10
    
    cw2 = 1
    cb2 = 1
    cw1 = 1
    cb1 = 1


    t1 = time.time()

    #batch
    for i in range(10):
        X_shuffle,Y_train_shuffle = shuffle(X_train,yindi_train)
        for ii in range(int(batch_num)):

            x_tem = X_shuffle[int(i*batch_size):int((i+1)*batch_size)]
            y_tem = Y_train_shuffle[int(i*batch_size):int((i+1)*batch_size)]

            y_fit, z = forward(x = x_tem, w1 = W1, b1 = b1, w2 = W2, b2 = b2, method = 'relu')

            # W2 -= eta*(deri_w2(z = z, y= y_fit,t = y_tem) + penalty * W2)
            # b2 -= eta*(deri_b2(y = y_fit, t = y_tem) + penalty*b2)
            # W1 -= eta*(deri_w1(X = x_tem,Z = z,T = y_tem, Y = y_fit, W2 = W2) + penalty*W1 )
            # b1 -= eta*(deri_b1(Z = z,T = y_tem, Y = y_fit,W2= W2) + penalty*b1)
            
            #the only new thing is the update rule
            dw2 = (deri_w2(z = z, y= y_fit,t = y_tem) + penalty * W2)
            db2 = (deri_b2(y = y_fit, t = y_tem) + penalty*b2)
            dw1 = (deri_w1(X = x_tem,Z = z,T = y_tem, Y = y_fit, W2 = W2) + penalty*W1 )
            db1 = (deri_b1(Z = z,T = y_tem, Y = y_fit,W2= W2) + penalty*b1) 

            cw2 = decay * cw2 + (1-decay)* dw2* dw2
            cb2 = decay * cb2 + (1-decay)* db2* db2
            cw1 = decay * cw1 + (1-decay)* dw1* dw1
            cb1 = decay * cb1 + (1-decay)* db1* db1

            W2 -= eta*dw2/(np.sqrt(cw2) + eps)
            b2 -= eta*db2/(np.sqrt(cb2) + eps)
            W1 -= eta*dw1/(np.sqrt(cw1) + eps)
            b1 -= eta*db1/(np.sqrt(cb1) + eps)



            p_y_test,_ = forward(x = X_test,w1 = W1, b1=b1,w2= W2, b2 = b2,method = 'relu')
            cost_test_tem = cost(y_matrix = p_y_test,t_matrix = yindi_test)
            cost_test.append(cost_test_tem)

            
        error_tem = error_rate(y_matrix = p_y_test, target = Y_test)
        print("the error rate in "+str(i)+"  is :"+str(error_tem))
    
    t2 = time.time()
    print("the whole process takes "+str(t2-t1)+" seconds")
    p_y_final,_ = forward(x = X_test,w1 = W1, b1=b1,w2= W2, b2 = b2,method = 'relu')
    error_final = error_rate(y_matrix = p_y_final, target = Y_test)
    print("the final error rate is "+str(error_final))

rmsprop()