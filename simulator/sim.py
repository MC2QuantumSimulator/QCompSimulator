import argparse
import numpy as np
from collections import Counter
import os
import random
import ParseInput
import circuitBuilder
from qmatrix import qmatrix
from qvector import qvector
import interpreter as interpreter

def main():
    parser = argparse.ArgumentParser(description='Tree based qasm sim.')

    # Assign a path to the arg 'gates'
    parser.add_argument('-g', dest='gates', help='File with gates', required=True)
    # Assign a path to the arg 'circuit'
    parser.add_argument('-c', dest='circuit', help='qasm file with circuit', required=True)
    # Assign a path to the arg 'input_state'
    parser.add_argument('-i', dest='input_state', help='File with input state(s)')
    group = parser.add_mutually_exclusive_group()
    # Numbers of shots to return
    group.add_argument('--shots', help='Return a number of bitstrings from final vector probabilities', type=int) #TODO
    # Wether to save the final vector or calculate probabilities
    group.add_argument('--state', action='store_true', help='Print the final vector to output.txt instead of calculating probabilities')
    # Save the matrix instead of calculating anything with it
    group.add_argument('--saveMatrix', help='Print the final matrix to path instead of multiplying with a vector')
    # Debug: print extra info
    parser.add_argument('--debug', action='store_true', help='Print extra debug info')
    args = parser.parse_args()

    gatespath = args.gates
    circuitpath = args.circuit
    inputstate = args.input_state
    shots = args.shots
    savestate = args.state
    save_matrix = args.saveMatrix
    debug = args.debug

    abs_output = os.path.join(os.path.dirname(__file__), "../outputFiles/output.txt")

    if debug:
        # Print out the input to see that it worked
        print('gatepath:    ' + gatespath)
        print('circuitpath: ' + circuitpath)
        print('input state: ' + ('No input, defaulting to |0>' if not inputstate else inputstate))
        if shots:
            print('num of reps: {}'.format(shots))
        print('Save state:  {}'.format(savestate))

    if debug:
        print("Starting Gates parsing")
    gate_names, gate_matrix = ParseInput.ParseInput.parse_gates(gatespath)
    if debug:
        print("Finished Gates parsing")
    if debug:
        print('\nList of all gates:')
        # pass the gates path to ParseInput ant print the returned lists
        for i, mat in enumerate(gate_matrix):
            print('{} = {}'.format(gate_names[i],mat))
        print('')


    if not shots: shots = 1024



    # Returns a matrix tree when passed a path to the qasm file and the parsed gates (or call parse gates from this func?)
    # Should circuitBuilder be a class that also includes measurment funcs and such or should that be in a different place?
   

    # Prints the resulting matrix in a Python/Matlab compatible format
    # TODO: Make it possible to output mathematical 'stuff' (exp, roots)
    if debug:
        print("Starting QASM parsing")
    qmat, height = interpreter.parse_qasm(circuitpath, gate_names, gate_matrix)
    if debug:
        print("Finished QASM parsing")

    if save_matrix:
        write_matrix_to_file(save_matrix, qmat)
    else:
        inputs = [] #
        outputstates = []

        if inputstate:
            f = open(inputstate, 'r')
            for state in eval(f.read()):
                inputs.append(state)
            f.close()
        else:
            input_vector = np.zeros(1<<height)
            input_vector[0] = 1
            inputs.append(input_vector)
           
        # The operation qmatrix applied to all input vectors
        for state in inputs:
            qv = qvector.to_tree(state)
            outputstates.append(qvector.mult(qmat,qv))
        
        # Mostly for debugging
        for output in outputstates:
            print("")
            print("Statevector: " + repr(output.to_vector()))
            print("Probability: " + repr(output.measure()))
        

        # Clears that "shots.txt" file, needed for multiple inputs
        abs_shots = os.path.join(os.path.dirname(__file__), "../outputFiles/shots.txt")
        f = open(abs_shots, 'w')
        f.write("")
        f.close()

        # Prints either state or output vectors
        if savestate:
            with open(abs_output, 'w') as f:
                f.write("State vectors: \n")
                for output in outputstates:
                    shoot(output.to_vector(),shots)
                    f.write(repr(output.to_vector()) + "\n")
        else:
            with open(abs_output, 'w') as f:
                f.write("Probabillity vectors: \n")
                for output in outputstates:
                    shoot(output.to_vector(),shots)
                    f.write(repr(output.measure()))
        

        

def write_matrix_to_file(save_matrix, qc):
    with open(save_matrix, 'w') as f:
        z = (1 << qc.height) - 1 # Last index of loop
        f.write('[')
        for x in range(1 << qc.height):
            if x != 0:
                f.write('\n')
            f.write('[')
            for y in range(1 << qc.height):
                string = ', ' if y != 0 else ''
                string += repr(round(qc.get_element((x, y)),7))
                f.write(string)
            f.write(']')
            if x != z:
                f.write(',')
        
        f.write(']')



def shoot(vector, reps):
    abs_shots = os.path.join(os.path.dirname(__file__), "../outputFiles/shots.txt")
    f = open(abs_shots, 'a')
    probs = [round(abs(x)**2,7) for x in vector]

    # Amount of bits
    n_bits = int(np.log2(len(vector)))
    
    # Array of all the collapsed bits
    bits = np.random.choice(
                            [np.binary_repr(x, n_bits) for x in np.arange(0,len(vector))]
                            ,reps,p=probs)

    # Shows statistics of how many times a value was chosen
    f.write(repr(Counter(bits)) + "\n")

    # Writes every collapsed bit
    for b in bits:
        f.write(b + "\n")


if __name__ == '__main__':
    main()