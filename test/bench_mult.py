import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import time
import numpy as np
import matplotlib.pyplot as plt
from simulator.qmatrix import qmatrix
from scipy.stats import linregress

def timerfunc(func):
    """
    A timer decorator
    """
    def function_timer(*args, **kwargs):
        """
        A nested function for timing other functions
        """
        start = time.time()
        value = func(*args, **kwargs)
        end = time.time()
        runtime = end - start
        return runtime, value
    return function_timer

@timerfunc
def idmult(n):
    id1 = qmatrix.id(n)
    id2 = qmatrix.id(n)
    res = qmatrix.mult(id1, id2)

nlist = []
times = []
for n in range(4,9):
    runtime = idmult(n)[0]
    print('n: {}, time: {}'.format(n,runtime))
    nlist.append(n)
    times.append(runtime)
logrun = np.log2(times)
test = linregress(nlist, logrun)
powlist = np.multiply(nlist,test[0])
print('O(2^{:.2f}*n), R = {}'.format(test[0], test[2]))
plt.plot(nlist, times, 'b')
plt.plot(nlist, np.float_power(2,powlist)*times[-1]/(2**powlist[-1]))
plt.xlabel('height of matrix')
plt.ylabel('runtime')
plt.show()