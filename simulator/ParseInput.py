import numpy as np
import math



class ParseInput:


    def parse_gates(test_file):
        f = open(test_file, 'r')
        gates_string_list = f.readlines()
        gate_names = []

        for gate in gates_string_list:
            gate = ''.join(gate.split()) #remove whitespaces
            split = gate.split('=')
            gate_names.append(split[0])
            split[1] #TODO how to parse the matrix part of gate







    if __name__ == '__main__':
        parse_gates('C:/Users/naomi/Documents/Kod projekt/PycharmProjects/inputFiles/gates.txt')


