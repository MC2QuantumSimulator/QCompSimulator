import qiskit as qs
import os
import math
import numpy as np
from qvector import qvector
from qmatrix import qmatrix


headers = ['OPENQASM', 'include']

# Creates a qvector of n qbits in the state 0
def qreg(n):
    vector = [0]*pow(2,n)
    vector[0] = 1
    
    return qvector.to_tree((np.array(vector)))

# Gets the int between '[]'
def get_int(split):
    return eval(split[1].split('[')[1].split(']')[0])


def parse_qasm(qasm_file):
    f_qasm = open(qasm_file, "r")
    qasm_string = f_qasm.readlines() 
    f_qasm.close()

    abs_gates = os.path.join(os.path.dirname(__file__), "../inputFiles/gates.txt")
    f_gate = open(abs_gates, "r")
    gates_string = f_gate.readlines()
    f_gate.close()

    gate_names = []
    gate_matrix = []
    
    # Appends gates' names and matrix'
    for line in gates_string:
        split = line.split('=')
        gate_names.append(split[0].strip())
        gate_matrix.append(eval(split[1].strip())) 
    
    #print(gate_names)
    #print(gate_matrix)

    variables = []
    operations = []

    q = None
    # Splits qasm into variable name and
    for line in qasm_string:
        split = line.split()
        var = split[0]

        # Creates quantum register of n bits
        if var == 'qreg':
            q = qreg(get_int(split))
        # If the gate is supported append var and op
        elif var in gate_names: 
            variables.append(split[0])
            operations.append(split[1])
        # Warns that the gate is not supported
        elif var not in headers:
            print(var + " is not supported")
            continue
    # Only works for whole vector and not single qbits
    # TODO: Få det att funka bitvis
    for var in variables:
        i = gate_names.index(var)
        mat = np.array(gate_matrix[i])
        #print(mat)
        size = int((len(qvector.to_vector(q)))/2) # Required size of identity matrix
        tree1 = qmatrix.to_tree(mat) # 
        tree2 = qmatrix.to_tree(np.identity(size))
        p = qmatrix.kron(tree1, tree2)
        q = qvector.mult(p, q) # TODO: Väldigt säker på att resultatet blir fel
    #print(variables)
    #print(operations)
    print(q.to_vector())

def gatepadding(gate: qmatrix, pre_n: int, tot_len: int) -> qmatrix:
    "Return a circuit layer created from a single gate"
    if pre_n > 0:
        # Prepend gate with correct sized identity matrix
        pre = qmatrix.id(pre_n)
        # Kron the trees together
        result_pre = qmatrix.kron(pre, gate)
    else:
        result_pre = gate

    # append identity matrix to fill up to same size as qreg
    post_n = tot_len-pre_n-gate.height
    if post_n > 0:
        post = qmatrix.id(tot_len-pre_n-gate.height)
        result = qmatrix.kron(result_pre, post)
    else:
        result = result_pre
    return result

if __name__ == '__main__':
    abs_qasm = os.path.join(os.path.dirname(__file__), "../inputFiles/qasm.txt")
    parse_qasm(abs_qasm)
    #q = qreg(3)
    #print(q)