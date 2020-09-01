# I have problem installing the theano package. So the codes below are not tested.

import theano
import theano.tensor as T
from theano.tensor.nnet import conv2d
from theano.tensor.signal import downsample
import numpy as np

def convpool(X, W, b, poolsize = (2,2)):
    conv_out = conv2d(input=X, filter = W)
    pool_out = downsample.max_pool_2d(
        input = copnv_out,
        ds = poolsize,
        ignore_border = True
    )

    return T.tanh(pool_out + b.dimshuffle('x',0,'x','x')) #dimshuffle is for broadcasting

def rearrange(X):
    #input is (32,32,3,N)
    #output is (N,3,32,32)
    N = X.shape[-1]
    out = np.zeros((N,3,32,32),dtype = np.float32)
    for i in range(N):
        for j in range(3):
            out[i,j,:,:] = X[:,:,j,i]
    return out/255

#set the weight
#convpool part
poolsize = 2
W1_shape = (20,3,5,5)
W1_init = filter_init(W1_shape, poolsize)
b1_init = np.zeros(W1_shape[0],dtype = np.float32)

W2_shape = (50,20,5,5)
W2_init = filter_init(W2_shape,poolsize)
b2_init = np.zeros(W2_shape[0],dtype = np.float32)

#ann part
W3_init = np.random.rand(W2_shape[0]*5*5,M)/np.sqrt(W2_shape[0]*5*5 + M)
b3_init = np.zeros(M, dtype = np.float32)
W4_init = np.random.rand(M,K)/np.sqrt(M+K)
b4_init = np.zeros(K, dtype = np.float32)

#forward pass
Z1 = convpool(X,W1,b1)
Z2 = convpool(Z1,W2,b2)
Z3 = relu(Z2.flatten(ndim=2).dot(W3)+b3) #ndim means flatten all dimensions after 2nd dimensions
pY = T.nnet.softmax(Z3.dot(W4) + b4)

