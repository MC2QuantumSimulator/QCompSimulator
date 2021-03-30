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

# Gets the index of the qbit between '[]'. 
# Returns '-1' if a gate should be applied to all qbits
def get_int(str):
    ev = str.split('[')

    # No single qbit defined, thus meaning gate should be applied to all qbits
    if len(ev) == 1: ev = -1
    # Extracts the index of the targeted qbit
    else: ev = eval(ev[1].split(']')[0])

    return ev



# Returns a list of qmatrixs to use on the qvector
def parse_qasm(qasm_file, gate_file):
    f_qasm = open(qasm_file, "r")
    qasm_string = f_qasm.readlines() 
    f_qasm.close()

    
    f_gate = open(gate_file, "r")
    gates_string = f_gate.readlines()
    f_gate.close()

    gate_names = []
    gate_matrix = []
    
    # Appends gates' names and matrix'
    for line in gates_string:
        split = line.split('=')
        gate_names.append(split[0].strip())
        gate_matrix.append(eval(split[1].strip())) 
    

    variables = []
    operations = []

    qmats = [] # List of qmatrixes to use on the qvector

    q = None
    # Splits qasm into variable name and
    for line in qasm_string:
        split = line.split()
        var = split[0]

        # Creates quantum register of n bits
        if var == 'qreg':
            q = qreg(get_int(split[1]))
        # If the gate is supported append var and op
        elif var in gate_names: 
            variables.append(split[0])
            operations.append(split[1])
        # Warns that the gate is not supported
        elif var not in headers:
            print(var + " is not supported")
            continue

    for index, var in enumerate(variables):

        ivar = gate_names.index(var)
        gate = qmatrix.to_tree(np.array(gate_matrix[ivar])) 
        qbit = get_int(operations[index])
        p = None
        temp = qmatrix.id(q.height)

        # Applies a gate to every qbit
        if qbit == -1: 
            for i in range(q.height):
                p = gatepadding(gate, i, q.height)
                temp = qmatrix.mult(temp, p)
                gate = qmatrix.to_tree(np.array(gate_matrix[ivar])) 

        # Applies a gate to a single qbit
        else:
            temp = gatepadding(gate, qbit, q.height)
        
        qmats.append(temp) # Adds qmatrix to later apply to qvector
        
    #for qmat in qmats:
    #    q = qvector.mult(qmat, q)
    #print(q.to_vector())
    
    return qmats, q.height

def gatepadding(gate: qmatrix, pre_n: int, tot_len: int) -> qmatrix:
    "Return a circuit layer created from a single gate"
    
    height = gate.height
    
    # Size of identity gate before gate represents which qbit the gate should be applied on
    # Identity matrix of size 2^n where n is the qbit the gate should be applied on
    if pre_n != 0:
        pre = qmatrix.id(pre_n)
        gate = qmatrix.kron(gate, pre)

    # Append identity matrix to fill up to same size as qreg
    if pre_n != tot_len-1:
        post = qmatrix.id(tot_len-pre_n-height)
        gate = qmatrix.kron(post, gate)
    
    return gate

        

if __name__ == '__main__':
    abs_qasm = os.path.join(os.path.dirname(__file__), "../inputFiles/qasm.txt")
    abs_gates = os.path.join(os.path.dirname(__file__), "../inputFiles/gates.txt")
    parse_qasm(abs_qasm, abs_gates)
    #q = qreg(3)
    #print(q)