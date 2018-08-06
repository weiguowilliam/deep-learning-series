import theano.tensor as T
from theano_ann import ANN
from util import get_spiral
import numpy as np

def grid_search():
    X, Y = get_spiral()
    