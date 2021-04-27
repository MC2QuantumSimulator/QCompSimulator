import qiskit as qs
import os
import math
import numpy as np
import sys
from qvector import qvector
from qmatrix import qmatrix
import ParseInput


headers = ['OPENQASM', 'include']



# Gets the index of the qbit between '[]'. 
# Returns '-1' if a gate should be applied to all qbits
def get_int(str):
    ev = str.split('[',1)

    # No single qbit defined, thus meaning gate should be applied to all qbits
    if len(ev) == 1: ev = -1
    # Extracts the index of the targeted qbit
    else: 
        ev = ev[1].split(']',1)

        # If it is a controlled operation return a touple of both qbits instead
        if ev[1][0] == ",":
            ev = (eval(ev[0]),get_int(ev[1]))
        else: ev = eval(ev[0])
   
    return ev



# Returns a list of qmatrixs to use on the qvector
def parse_qasm(qasm_file, gate_names, gate_matrix):
    f_qasm = open(qasm_file, "r")
    qasm_string = f_qasm.readlines() 
    f_qasm.close()

    variables = []
    operations = []
    for x in range(3,1,-1): print(x)
    print("hej")
    for x in range(1,3): print(x)
    print("då")

    q = None
    # Splits qasm into variable name and
    for line in qasm_string:
        
        # Ignores empty lines
        if line.strip() == "": continue

        split = line.split()
        var = split[0]

        # Creates quantum register of n bits
        if var == 'qreg':
            height = get_int(split[1])
            qmat = qmatrix.id(height) # The qmatrix that will acumulate all the operations
        # If the gate is supported append var and op
        elif var in gate_names: 
            variables.append(split[0])
            operations.append(split[1])
        # Warns that the gate is not supported
        elif var not in headers:
            sys.exit("Error: " + var + " is not supported")
            

    for index, var in enumerate(variables):

        ivar = gate_names.index(var)
        gate = qmatrix.to_tree(np.array(gate_matrix[ivar])) 
        qbit = get_int(operations[index])
        
        
        # Applies a gate to every q bit
        if qbit == -1: 
            for i in range(height):
                q = gatepadding(gate, i, height)
                qmat = qmatrix.mult(q,qmat)
                gate = qmatrix.to_tree(np.array(gate_matrix[ivar])) 
        
        # For controlled operations
        elif isinstance(qbit, tuple):
            a = qbit[0]
            b = qbit[1]
            if a == b: sys.exit("Error: Controlled operation can't be applied to the same qbit")
            
            swap = qmatrix.to_tree(np.array(gate_matrix[gate_names.index("SWAP")]))


            if a < b:

                # Swaps qbit 'a' to neighbour bit 'b'
                for i in range(a,b-1):
                    q = gatepadding(swap, i, height)
                    qmat = qmatrix.mult(q,qmat)
                    swap = qmatrix.to_tree(np.array(gate_matrix[gate_names.index("SWAP")]))

                # Performs the controlled operation before swapping back position
                q = gatepadding(gate, b-1, height)
                qmat = qmatrix.mult(q,qmat)

                # Swaps back qbit 'a' to its original position
                for i in range(b-2,a-1,-1):
                    q = gatepadding(swap, i, height)
                    qmat = qmatrix.mult(q,qmat)
                    swap = qmatrix.to_tree(np.array(gate_matrix[gate_names.index("SWAP")]))
            
            else:
                # Swaps qbit 'b' to neighbour bit 'a'
                for i in range(a-1,b-1,-1):
                    q = gatepadding(swap, i, height)
                    qmat = qmatrix.mult(q,qmat)
                    swap = qmatrix.to_tree(np.array(gate_matrix[gate_names.index("SWAP")]))

                # Performs the controlled operation before swapping back position
                q = gatepadding(gate, b, height)
                qmat = qmatrix.mult(q,qmat)

                # Swaps back qbit 'b' and 'a' to its original position
                for i in range(b,a):
                    q = gatepadding(swap, i, height)
                    qmat = qmatrix.mult(q,qmat)
                    swap = qmatrix.to_tree(np.array(gate_matrix[gate_names.index("SWAP")]))





        # Applies a gate to a single qbit
        else:
            #print(type(qbit))
            q = gatepadding(gate, qbit, height)
            qmat = qmatrix.mult(q,qmat)
        

    return qmat, height

def gatepadding(gate: qmatrix, pre_n: int, tot_len: int) -> qmatrix:
    "Return a circuit layer created from a single gate"
    
    height = gate.height
    
    # Size of identity gate before gate represents which qbit the gate should be applied on
    # Identity matrix of size 2^n where n is the qbit the gate should be applied on
    if pre_n != 0:
        pre = qmatrix.id(pre_n)
        gate = qmatrix.kron(pre, gate)

    # Append identity matrix to fill up to same size as qreg
    if pre_n != tot_len-height:
        post = qmatrix.id(tot_len-pre_n-height)
        gate = qmatrix.kron(gate, post)
    
    #print(gate.to_matrix())
    return gate

        

if __name__ == '__main__':
    abs_qasm = os.path.join(os.path.dirname(__file__), "../inputFiles/qasm.txt")
    abs_gates = os.path.join(os.path.dirname(__file__), "../inputFiles/gates.txt")
    gate_names, gate_matrix = ParseInput.ParseInput.parse_gates(abs_gates) 
    parse_qasm(abs_qasm, gate_names, gate_matrix)
    