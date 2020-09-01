import numpy as np
from sklearn.utils import shuffle
from util import get_transformed_digit,  error_rate, cost,y2indicator
from mlp import forward, deri_w1, deri_w2, deri_b1, deri_b2
import time

def adam():
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
    
    penalty = 0.001

    batch_size = 500
    batch_num = N // batch_size

    # two new variable
    decay = 0.9
    eps = 1e-10

    eta = 1e-3
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    
    cw2 = 1
    cb2 = 1
    cw1 = 1
    cb1 = 1

    mw2 = 0
    mb2 = 0
    mw1 = 0
    mb1 = 0

    vw2 = 0
    vb2 = 0
    vw1 = 0
    vb1 = 0

    t = 1


    t1 = time.time()

    #batch
    for i in range(10):
        X_shuffle,Y_train_shuffle = shuffle(X_train,yindi_train)
        for ii in range(int(batch_num)):

            x_tem = X_shuffle[int(i*batch_size):int((i+1)*batch_size)]
            y_tem = Y_train_shuffle[int(i*batch_size):int((i+1)*batch_size)]

            y_fit, z = forward(x = x_tem, w1 = W1, b1 = b1, w2 = W2, b2 = b2, method = 'relu')

####################### adam ##############################

            gW2 = deri_w2(z, y_fit, y_tem) + penalty*W2
            gb2 = deri_b2(y_fit, y_tem) + penalty*b2
            gW1 = deri_w1(x_tem, z, y_tem, y_fit, W2) + penalty*W1
            gb1 = deri_b1(z, y_tem, y_fit, W2) + penalty*b1

            # new m
            mw1 = beta1 * mw1 + (1 - beta1) * gW1
            mb1 = beta1 * mb1 + (1 - beta1) * gb1
            mw2 = beta1 * mw2 + (1 - beta1) * gW2
            mb2 = beta1 * mb2 + (1 - beta1) * gb2

            # new v
            vw1 = beta2 * vw1 + (1 - beta2) * gW1 * gW1
            vb1 = beta2 * vb1 + (1 - beta2) * gb1 * gb1
            vw2 = beta2 * vw2 + (1 - beta2) * gW2 * gW2
            vb2 = beta2 * vb2 + (1 - beta2) * gb2 * gb2

            # bias correction
            correction1 = 1 - beta1 ** t
            hat_mw1 = mw1 / correction1
            hat_mb1 = mb1 / correction1
            hat_mw2 = mw2 / correction1
            hat_mb2 = mb2 / correction1


            correction2 = 1 - beta2 ** t
            hat_vw1 = vw1 / correction2
            hat_vb1 = vb1 / correction2
            hat_vw2 = vw2 / correction2
            hat_vb2 = vb2 / correction2

            # update t
            t += 1

            # apply updates to the params
            W1 = W1 - eta * hat_mw1 / np.sqrt(hat_vw1 + eps)
            b1 = b1 - eta * hat_mb1 / np.sqrt(hat_vb1 + eps)
            W2 = W2 - eta * hat_mw2 / np.sqrt(hat_vw2 + eps)
            b2 = b2 - eta * hat_mb2 / np.sqrt(hat_vb2 + eps)

##########################################################

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

adam()