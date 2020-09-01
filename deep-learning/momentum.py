import numpy as np
from sklearn.utils import shuffle
from util import get_transformed_digit,  error_rate, cost,y2indicator
from mlp import forward, deri_w1, deri_w2, deri_b1, deri_b2
import time

def benchmark_batch():
    """
    use util functions to run the logistic classification with bp
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


    t1 = time.time()

    #batch
    for i in range(100):
        X_shuffle,Y_train_shuffle = shuffle(X_train,yindi_train)
        for ii in range(int(batch_num)):
            # x_tem = X_shuffle[ii].reshape(1,D)
            # y_tem = Y_train_shuffle[ii].reshape(1,10)

            x_tem = X_shuffle[int(i*batch_size):int((i+1)*batch_size)]
            y_tem = Y_train_shuffle[int(i*batch_size):int((i+1)*batch_size)]

            # y_fit = forward(x = x_tem,w=W,b=b)
            y_fit, z = forward(x = x_tem, w1 = W1, b1 = b1, w2 = W2, b2 = b2, method = 'relu')

            W2 -= eta*(deri_w2(z = z, y= y_fit,t = y_tem) + penalty * W2)
            b2 -= eta*(deri_b2(y = y_fit, t = y_tem) + penalty*b2)
            W1 -= eta*(deri_w1(X = x_tem,Z = z,T = y_tem, Y = y_fit, W2 = W2) + penalty*W1 )
            b1 -= eta*(deri_b1(Z = z,T = y_tem, Y = y_fit,W2= W2) + penalty*b1)
            # W2 -= eta*(deri_w2(z = z, y= y_fit,t = y_tem) )
            # b2 -= eta*(deri_b2(y = y_fit, t = y_tem) )
            # W1 -= eta*(deri_w1(X = x_tem,Z = z,T = y_tem, Y = y_fit, W2 = W2) )
            # b1 -= eta*(deri_b1(Z = z,T = y_tem, Y = y_fit,W2= W2))

            
            # W += eta*(deri_w(t_matrix = y_tem, y_matrix = y_fit,x = x_tem)-penalty*W)
            # b += eta*(deri_b(t_matrix = y_tem, y_matrix = y_fit)-penalty*b)

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


def momentum_batch():
    """
    use util functions to run the logistic classification with bp
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

    mu = 0.9



    vw2 = 0
    vb2 = 0
    vw1 = 0
    vb1 = 0


    t1 = time.time()

    #batch
    for i in range(100):
        X_shuffle,Y_train_shuffle = shuffle(X_train,yindi_train)
        for ii in range(int(batch_num)):
            # x_tem = X_shuffle[ii].reshape(1,D)
            # y_tem = Y_train_shuffle[ii].reshape(1,10)

            x_tem = X_shuffle[int(i*batch_size):int((i+1)*batch_size)]
            y_tem = Y_train_shuffle[int(i*batch_size):int((i+1)*batch_size)]

            # y_fit = forward(x = x_tem,w=W,b=b)
            y_fit, z = forward(x = x_tem, w1 = W1, b1 = b1, w2 = W2, b2 = b2, method = 'relu')

            #the only change to benchmark batch is the update rule:
            gw2 = deri_w2(z = z, y= y_fit,t = y_tem) + penalty * W2
            gb2 = deri_b2(y = y_fit, t = y_tem) + penalty*b2
            gw1 = deri_w1(X = x_tem,Z = z,T = y_tem, Y = y_fit, W2 = W2) + penalty*W1
            gb1 = eta*(deri_b1(Z = z,T = y_tem, Y = y_fit,W2= W2) + penalty*b1)

            vw2 = mu*vw2 - eta * gw2
            vb2 = mu*vb2 - eta * gb2
            vw1 = mu*vw1 - eta * gw1
            vb1 = mu*vb1 - eta * gb1

            W2 += vw2
            b2 += vb2
            W1 += vw1
            b1 += vb1


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


def nesterov_momentum_batch():
    """
    use util functions to run the logistic classification with bp
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

    mu = 0.9



    vw2 = 0
    vb2 = 0
    vw1 = 0
    vb1 = 0


    t1 = time.time()

    #batch
    for i in range(100):
        X_shuffle,Y_train_shuffle = shuffle(X_train,yindi_train)
        for ii in range(int(batch_num)):
            # x_tem = X_shuffle[ii].reshape(1,D)
            # y_tem = Y_train_shuffle[ii].reshape(1,10)

            x_tem = X_shuffle[int(i*batch_size):int((i+1)*batch_size)]
            y_tem = Y_train_shuffle[int(i*batch_size):int((i+1)*batch_size)]

            # y_fit = forward(x = x_tem,w=W,b=b)
            y_fit, z = forward(x = x_tem, w1 = W1, b1 = b1, w2 = W2, b2 = b2, method = 'relu')

            #the only change to benchmark batch is the update rule:
            gw2 = deri_w2(z = z, y= y_fit,t = y_tem) + penalty * W2
            gb2 = deri_b2(y = y_fit, t = y_tem) + penalty*b2
            gw1 = deri_w1(X = x_tem,Z = z,T = y_tem, Y = y_fit, W2 = W2) + penalty*W1
            gb1 = eta*(deri_b1(Z = z,T = y_tem, Y = y_fit,W2= W2) + penalty*b1)

            vw2 = mu*vw2 - eta * gw2
            vb2 = mu*vb2 - eta * gb2
            vw1 = mu*vw1 - eta * gw1
            vb1 = mu*vb1 - eta * gb1

            W2 += mu*vw2 - eta * gw2
            b2 += mu*vb2 - eta * gb2
            W1 += mu*vw1 - eta * gw1
            b1 += mu*vb1 - eta * gb1


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

nesterov_momentum_batch()