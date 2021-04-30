import sim
import time
import testQiskit
import numpy as np
from scipy.stats import linregress
import matplotlib.pyplot as plt
import profile
import gc

gatespath = "inputFiles/qftgates.txt"
circuitpath = "inputFiles/testQASMprint.txt"


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
def run_our_sim(gatespath, circuitpath):
	sim.main(gatespath, circuitpath, debug=False, inputstate=None, save_matrix=None, qiskit_ordering=None, shots=None, savestate=None)

text = testQiskit.create_qft(6)
with open("inputFiles/testQASMprint.txt", "w") as f:
	f.write(text)
#profile.run('run_our_sim(gatespath, circuitpath)', sort='tottime')
nlist = []
times = []
for n in range(3,8):
	gc.collect()
	text = testQiskit.create_qft(n)
	with open("inputFiles/testQASMprint.txt", "w") as f:
		f.write(text)
	runtime, res = run_our_sim(gatespath, circuitpath)
	print(runtime)
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