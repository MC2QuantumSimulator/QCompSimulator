import sys

import numpy as np
import os
import math
import fourFn

# Checks if a matrix is unitary
def is_unitary(matrix):
        return np.allclose(np.eye(len(matrix)), matrix * matrix.H)

class ParseInput:


    def parse_gates(input_file):
        
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
            dimension = eval(gate_size.pop(0))
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

        lines = []
        for index in range(len(gate_names)):
            lines.append(gate_names[index] + " = " + repr(gates_matrix_form[index]))

        return gate_names, gates_matrix_form




    if __name__ == '__main__':
        abs_input = os.path.join(os.path.dirname(__file__), "../inputFiles/gates.txt") # Always gives the correct path (atleast for Linux)
        
        
        gates = parse_gates(abs_input)
        
        #output_gates(abs_output, gates)
       

            