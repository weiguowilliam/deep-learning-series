import theano 
import theano.tensor as T
import numpy as np
import time
import matplotlib.pyplot as plt
from theano.tensor.nnet import conv2d
# from theano.tensor.signal import downsample
from theano.tensor.signal import pool
from scipy.io import loadmat
from sklearn.utils import shuffle

import time

def error_rate(p,y):
    return np.mean(p!=y)

def relu(a):
    return a * (a>0)

def y2indicator(Y):
    N = len(Y)
    ind = np.zeros((N,10))
    for i in range(N):
        ind[i,Y[i]] = 1
    return ind

def conv_pool(X,W,b,poolsize=(2,2)):
    conv_out = conv2d(input = X, filters = W)
    pool_out = pool.pool_2d(
        input = conv_out,
        ws = poolsize,
        ignore_border = True
    )
    return T.tanh(pool_out + b.dimshuffle('x',0,'x','x'))

def init_filter(shape, poolsize):
    w = np.random.rand(*shape)/np.sqrt(np.prod(shape[1:])) + shape[0]*np.prod(shape[2:]/np.prod(poolsize))
    return w.astype(np.float32)

def rearrange(X):
    N = X.shape[-1]
    out = np.zeros((N,3,32,32),dtype = np.float32)
    for i in range(N):
        for j in range(3):
            out[i,j,:,:] = X[:,:,j,i]
    return out

def cnn_theano():
    #import
    train = loadmat('C:/Users/Wei Guo/Desktop/deep-learning-series-lesson/data/train_32x32.mat')
    test = loadmat('C:/Users/Wei Guo/Desktop/deep-learning-series-lesson/data/test_32x32.mat')
    
    #split data
    X_train = rearrange(train['X'])
    Y_train = train['y'].flatten() - 1 
    del train
    X_train, Y_train = shuffle(X_train, Y_train)
    Y_train_indi = y2indicator(Y_train)

    X_test = rearrange(test['X'])
    Y_test = test['y'].flatten() - 1
    del test
    Y_test_indi = y2indicator(Y_test)

    lr = np.float32(0.0001)
    reg = np.float32(0.01)
    mu = np.float32(0.99)

    N = X_train.shape[0]
    batch_size = 500
    batch_num = N//batch_size

    M = 500
    K = 10
    poolsize = (2,2)

    # convpool part
    W1_shape = (20,3,5,5)
    W1_init = init_filter(W1_shape,poolsize)
    b1_init = np.zeros(W1_shape[0],np.float32)

    W2_shape = (50,20,5,5)
    W2_init = init_filter(W2_shape,poolsize)
    b2_init = np.zeros(W2_shape[0],np.float32)

    #ann part
    W3_init = np.random.rand(W2_shape[0]*5*5,M)/np.sqrt(W2_shape[0]*5*5 + M)
    b3_init = np.zeros(M, dtype = np.float32)

    W4_init = np.random.rand(50,10)/np.sqrt(50+10)
    b4_init = np.zeros(10,dtype = np.float32)

    # theano part
    X = T.tensor4('X',dtype='float32')
    Y = T.matrix('T')

    W1 = theano.shared(W1_init,'W1')
    b1 = theano.shared(b1_init,'b1')
    W2 = theano.shared(W2_init,'W2')
    b2 = theano.shared(b2_init,'b2')
    W3 = theano.shared(W3_init.astype(np.float32),'W3')
    b3 = theano.shared(b3_init,'b3')
    W4 = theano.shared(W4_init.astype(np.float32),'W4')
    b4 = theano.shared(b4_init,'b4')

    dw1 = theano.shared(np.zeros(W1_init.shape,dtype=np.float32),'dw1')
    db1 = theano.shared(np.zeros(b1_init.shape,dtype=np.float32),'db1')
    dw2 = theano.shared(np.zeros(W2_init.shape,dtype=np.float32),'dw2')
    db2 = theano.shared(np.zeros(b2_init.shape,dtype=np.float32),'db2')
    dw3 = theano.shared(np.zeros(W3_init.shape,dtype=np.float32),'dw3')
    db3 = theano.shared(np.zeros(b3_init.shape,dtype=np.float32),'db3')
    dw4 = theano.shared(np.zeros(W4_init.shape,dtype=np.float32),'dw4')
    db4 = theano.shared(np.zeros(b4_init.shape,dtype=np.float32),'db4')

    #forward
    Z1 = conv_pool(X,W1,b1)
    Z2 = conv_pool(Z1,W2,b2)
    Z3 = relu(Z2.flatten(ndim=2).dot(W3) + b3)
    pY = T.nnet.softmax(Z3.dot(W4) + b4)

    params = (W1,b1,W2,b2,W3,b3,W4,b4)
    reg_cost = reg*np.sum((param*param).sum() for param in params)

    cost = -(Y*T.log(pY)).sum() + reg_cost
    prediction = T.argmax(pY,axis = 1)

    #update method: momentum
    update_w1 = W1 + mu*dw1 - lr*T.grad(cost,W1)
    update_b1 = b1 + mu*db1 - lr*T.grad(cost,b1)
    update_w2 = W2 + mu*dw2 - lr*T.grad(cost,W2)
    update_b2 = b2 + mu*db2 - lr*T.grad(cost,b2)
    update_w3 = W3 + mu*dw3 - lr*T.grad(cost,W3)
    update_b3 = b3 + mu*db3 - lr*T.grad(cost,b3)
    update_w4 = W4 + mu*dw4 - lr*T.grad(cost,W4)
    update_b4 = b4 + mu*db4 - lr*T.grad(cost,b4)

    update_dw1 = mu*dw1 - lr*T.grad(cost,W1)
    update_db1 = mu*db1 - lr*T.grad(cost,b1)
    update_dw2 = mu*dw2 - lr*T.grad(cost,W2)
    update_db2 = mu*db2 - lr*T.grad(cost,b2)
    update_dw3 = mu*dw3 - lr*T.grad(cost,W3)
    update_db3 = mu*db3 - lr*T.grad(cost,b3)
    update_dw4 = mu*dw4 - lr*T.grad(cost,W4)
    update_db4 = mu*db4 - lr*T.grad(cost,b4)

    train = theano.function(
        inputs = [X,Y],
        updates = [
            (W1,update_w1),
            (b1,update_b1),
            (W2,update_w2),
            (b2,update_b2),
            (W3,update_w3),
            (b3,update_b3),
            (W4,update_w4),
            (b4,update_b4),
            (dw1,update_dw1),
            (db1,update_db1),
            (dw2,update_dw2),
            (db2,update_db2),
            (dw3,update_dw3),
            (db3,update_db3),
            (dw4,update_dw4),
            (db4,update_db4),
        ]
    )

    get_prediction = theano.function(
        inputs = [X,Y],
        outputs = [cost, prediction],
    )

    time0 = time.time()
    LL = []
    for i in range(20):
        for j in range(batch_num):
            Xbatch = X_train[j*batch_size:(j+1)*batch_size,]
            Ybatch = Y_train_indi[j*batch_size:(j+1)*batch_size,]

            train(Xbatch,Ybatch)

            if j % 50 == 0:
                cost_val,prediction_val = get_prediction(X_test,Y_test_indi)
                err = error_rate(p=prediction_val,y=Y_test)
                LL.append(cost_val)
                print("cost/err is "+str(cost_val)+" "+ str(err)+" at the round "+ str(i)+" "+str(j))
    cost_final,prediction_final = get_prediction(X_test,Y_test_indi)
    err_final = error_rate(p=prediction_final,y=Y_test)
    print("the final error rate is "+ str(err_final))
    elap = time.time() - time0
    print("the time elapsed is "+str(elap)+ " seconds")
    plt.plot(LL)
    plt.show()



cnn_theano()


















