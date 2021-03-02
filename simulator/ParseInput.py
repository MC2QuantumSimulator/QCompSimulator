import sys

import numpy as np
import math
import fourFn


class ParseInput:

    def parse_gates(test_file):

        f = open(test_file, 'r')
        gates_string_list = f.readlines()
        gates_string_form = []
        gate_names = []
        gates_matrix_form = []
        gate_size = []

        # split the matrices into separate lists containing, name, elements and dimension
        for gate in gates_string_list:
            gate = "".join(gate.split())
            split = gate.split('=')
            gate_names.append(split[0])
            gates_string_form.append(split[1])
            gate_size.append(split[2])

        for gate in gates_string_form:
            dimension = eval(gate_size.pop())
            Matrix = [[0 for x in range(dimension)] for y in range(dimension)]
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
                    Matrix[i][j] = fourFn.eval(elements.pop(0))
                    # TODO är det en unitär matris?
            gates_matrix_form.append(Matrix)

        return gates_matrix_form

    if __name__ == '__main__':
        g = parse_gates('C:/Users/naomi/Documents/Kod projekt/PycharmProjects/inputFiles/gates.txt')
        # ----------for debugging-------
        for gate in g:
            print(gate)
