import sys

import numpy as np
import os
import math
import fourFn

# Checks if a matrix is unitary
def is_unitary(matrix):
        return np.allclose(np.eye(len(matrix)), matrix * matrix.H)

class ParseInput:

#TODO error messages on which specific line
#TODO add Global wieght
#TODO add support for the trigonometric functions
#TODO Add bracket syntax
#TODO delete the size column in input

    def parse_gates(input_file):
        
        f = open(input_file, 'r')
        gates_string_list = f.readlines()
        f.close()

        gates_string_form = []
        gate_names = []
        gates_matrix_form = []
        not_unitary = []

        # Split the matrices into separate lists containing, name and elements
        for gate in gates_string_list:
            gate = "".join(gate.split())
            split = gate.split('=')
            gate_names.append(split[0])
            gates_string_form.append(split[1])

        # Go over each gate and turn into a matrix if it's unitary.
        for index, gate in enumerate(gates_string_form):
            #split  and trim global weight and gate definition
            split = gate.split('[')
            gate = split[1].replace(']', '')

            # Splitting into a list containing all elements in string form
            rows = gate.split(';')
            elements = []
            for row in rows:
                tmp = row.strip()
                tmp = tmp.split(',')
                elements = elements + tmp

            # Create the matrix
            dimension = len(rows)
            matrix = [[0 for _ in range(dimension)] for _ in range(dimension)]

            #Check for global variable
            if split[0] == '':
                # Put elements in matrix
                for i in range(dimension):
                    for j in range(dimension):
                        matrix[i][j] = fourFn.eval(elements.pop(0))
            else:
                global_var = split[0].strip('*')
                global_var = fourFn.eval(global_var)

                # Put elements in matrix
                for i in range(dimension):
                    for j in range(dimension):
                        matrix[i][j] = fourFn.eval(elements.pop(0))*global_var

            # Checks if matrix is unitary and adds it to 'gates_matrix_form' if so        
            if not is_unitary(np.matrix(matrix)):
                print(gate_names[index] + ": is not unitary")
                #gate_names.pop(index)
                not_unitary.append(index)
                continue

            gates_matrix_form.append(matrix)
            print(matrix)
        
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
       

            