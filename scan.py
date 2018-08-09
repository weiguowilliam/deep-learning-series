import numpy as np
import theano
import theano.tensor as T

x= T.vector('x')

def squ(x):
    return x**2

outputs,_ = theano.scan(
    fn = squ,
    sequences = x,
    n_steps = x.shape[0]
)

squ_op = theano.function(
    inputs=[x],
    outputs = [outputs]
)

output_eval = squ_op(np.array([1,2,3]))
print(output_eval)

