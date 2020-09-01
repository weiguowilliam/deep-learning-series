# here are the common functions used in mlp. Some functions have the same name as those
# in util.py(forward function)
import numpy as np

def forward(x,w1,b1,w2,b2,method):
    if method == 'sigmoid':
        tem = x.dot(w1)+b1
        z = 1/(1+np.exp((-1)*tem))
    if method == 'relu':
        z = x.dot(w1) + b1
        z[z<0] = 0
    else:
        print("input the correct method")
        return None
    
    a = z.dot(w2) + b2
    expa = np.exp(a)
    y = expa/np.sum(expa,axis=1,keepdims=True)
    
    return y,z
    
def deri_w2(z,y,t):
    return z.T.dot(y - t)

def deri_b2(y,t):
    return np.sum(y-t, axis = 0)

def deri_w1(X, Z, T, Y, W2):
    # return X.T.dot( ( ( Y-T ).dot(W2.T) * ( Z*(1 - Z) ) ) ) # for sigmoid
    return X.T.dot( ( ( Y-T ).dot(W2.T) * (Z > 0) ) ) # for relu

def deri_b1(Z, T, Y, W2):
    # return (( Y-T ).dot(W2.T) * ( Z*(1 - Z) )).sum(axis=0) # for sigmoid
    return (( Y-T ).dot(W2.T) * (Z > 0)).sum(axis=0) # for relu

