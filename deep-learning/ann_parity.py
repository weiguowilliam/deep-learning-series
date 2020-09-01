import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

## two independent function
def init_weight(a,b):
    return np.random.rand(a,b)/np.sqrt(a+b)

# just copy the function- -
def all_parity_pairs(nbit):
    # total number of samples (Ntotal) will be a multiple of 100
    # why did I make it this way? I don't remember.
    N = 2**nbit
    remainder = 100 - (N % 100)
    Ntotal = N + remainder
    X = np.zeros((Ntotal, nbit))
    Y = np.zeros(Ntotal)
    for ii in range(Ntotal):
        i = ii % N
        # now generate the ith sample
        for j in range(nbit):
            if i % (2**(j+1)) != 0:
                i -= 2**j
                X[ii,j] = 1
        Y[ii] = X[ii].sum() % 2
    return X, Y

class HiddenLayer(object):
    def __init__(self,m1,m2,id):
        self.id = id
        self.m1 = m1
        self.m2 = m2
        W = init_weight(m1,m2)
        b = np.zeros(m2)
        self.W = theano.shared(W,'W%s'%(self.id))
        self.b = theano.shared(b,'b%s'%(self.id))
        self.params = [self.W, self.b]
    
    def forward(self,X):
        return T.nnet.relu(X.dot(self.W) + self.b)

class ANN(object):
    def __init__(self,hidden_layer_size):
        self.hidden_layer_size = hidden_layer_size
    
    
    def forward(self,X):
        Z = X
        for h in self.hidden_layers:
            Z = h.forward(Z)
        return T.nnet.softmax(Z.dot(self.W)+self.b) #w,b used here are the w,b in last step(z to y)
    
    def predict(self,X):
        pY = self.forward(X)
        return T.argmax(pY, axis = 1)

    def fit(self,X,Y,lr = 0.00001, mu = 0.99,reg = 1e-12, epi = 200,batch_size = 50, print_period = 1, show_plt = False):
          
        #retype X,Y
        X = X.astype(np.float32)
        Y = Y.astype(np.float32)

        #init the w
        N, D = X.shape
        K = Y.shape[0]

        self.hidden_layers = []

        m1 = D
        count = 0

        for m in self.hidden_layer_size:
            hidden_layer = HiddenLayer(m1 = m1, m2 = m,id = count)
            count += 1
            self.hidden_layers.append(hidden_layer)
            m1 = m
        
        W_last = init_weight(a = m1, b = K)
        b_last = np.zeros(K)
        self.W = theano.shared(W_last,'W_last')
        self.b = theano.shared(b_last,'b_last')

        # collect params
        self.params = [self.W, self.b]
        for param in self.hidden_layers:
            self.params += param.params
        
        # prepare for momentum and rmsprot
        dparams = [theano.shared(np.zeros(p.get_value().shape)) for p in self.params]
        cache   = [theano.shared(np.zeros(p.get_value().shape)) for p in self.params]

        #set up for theano
        thX = T.matrix('X')
        thY = T.ivector('Y')
        pY = self.forward(X = thX)

        rcost = reg*T.sum([(p*p).sum() for p in self.params])
        cost = -T.mean(T.log(pY[T.arange(thY.shape[0]),thY])) + rcost
        prediction = self.predict(thX)
        grads = T.grad(cost = cost, wrt = self.params)

        # momentum updates
        updates = [
            (p, p + mu*dp - lr*g) for (p, dp, g) in zip(self.params, dparams, grads) 
            ]+[
                (dp, mu*dp - lr*g) for (dp,g) in zip(dparams, grads)
            ]
        
        train_op = theano.function(
            inputs = [thX,thY],
            outputs = [cost,prediction],
            updates = updates,
        )

        batch_num = N//batch_size

        costs = []

        for i in range(epi):
            X, Y = shuffle(X,Y)
            for j in range(batch_num):
                X_batch = X[j*batch_size:(j+1)*batch_size]
                Y_batch = Y[j*batch_size:(j+1)*batch_size]
                c,p = train_op(X_batch,Y_batch)

                if j%20 == 0:
                    costs.append(c)
                    err_rate = np.mean(p != Y_batch)
                    print('the error rate is %s'%(err_rate))
        
        if show_plt == True:
            plt.plot(costs)
            plt.show()

def wide():
    X,Y = all_parity_pairs(12)
    ann = ANN([2048])
    ann.fit(X,Y,show_plt=True)
    
def deep():
    X,Y = all_parity_pairs(12)
    ann = ANN([1024,1024])
    ann.fit(X,Y,show_plt=True)

wide()

        

            

