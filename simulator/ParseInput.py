import sys

import numpy as np
import os
import math
import fourFn

# Checks if a matrix is unitary
def is_unitary(matrix):
        return np.allclose(np.eye(len(matrix)), matrix * matrix.H)

class ParseInput:


    def input_gates(input_file):
        
        f = open(input_file, 'r')
        gates_string_list = f.readlines()
        f.close()
        gates_string_form = []
        gate_names = []
        gates_matrix_form = []
        gate_size = []
        not_unitary = []



        # Split the matrices into separate lists containing, name, elements and dimension
        for gate in gates_string_list:
            gate = "".join(gate.split())
            split = gate.split('=')
            gate_names.append(split[0])
            gates_string_form.append(split[1])
            gate_size.append(split[2])

        for index ,gate in enumerate(gates_string_form):
            dimension = eval(gate_size.pop())
            matrix = [[0 for _ in range(dimension)] for _ in range(dimension)]
            rows = gate.split(';')

            # Splitting into a list containing all elements in string form
            elements = []
            for row in rows:
                tmp = row.strip()
                tmp = tmp.split(',')
                elements = elements + tmp

            # Put elements in matrix
            for i in range(dimension):
                for j in range(dimension):
                    matrix[i][j] = fourFn.eval(elements.pop(0))

            # Checks if matrix is unitary and adds it to 'gates_matrix_form' if so        
            if not is_unitary(np.matrix(matrix)):
                print(gate_names[index] + ": is not unitary")
                #gate_names.pop(index)
                not_unitary.append(index)
                continue

            gates_matrix_form.append(matrix)
        
        # Deletes the non-unitary gate names.
        for index in range(len(not_unitary)-1,-1,-1):
            gate_names.pop(not_unitary[index])

        return gate_names, gates_matrix_form

    
    def output_gates(output_file, input_gates):

        f = open(output_file, 'r')
        gates_existing = f.readlines()
        f.close()
        f = open(output_file, 'a')
        gates_names = []

        # Collects the name of existing gates
        for gate in gates_existing:
            gates_names.append(gate.split()[0])

        # Adds a gate if it doesn't
        for index in range(len(input_gates[0])):
            if input_gates[0][index] in gates_names: continue
            line = input_gates[0][index] + " = " + repr(input_gates[1][index]) + "\n"
            f.write(line)
            
            



    if __name__ == '__main__':
        abs_input = os.path.join(os.path.dirname(__file__), "../inputFiles/gates_input.txt") # Always gives the correct path (atleast for Linux)
        abs_output = os.path.join(os.path.dirname(__file__), "../inputFiles/gates.txt")
        
        gates = input_gates(abs_input)
        
        output_gates(abs_output, gates)
       

            