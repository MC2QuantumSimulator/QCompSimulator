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

def main(gatespath, circuitpath, inputstate, shots, savestate, save_matrix, qiskit_ordering, debug, test):

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
            if qiskit_ordering:
                print("State vector: " + repr([round(x,7) for x in swap_significants(output.to_vector())]))
                print("Probability: " + repr([round(x,7) for x in swap_significants(output.measure())]))
            else:
                print("State vector: " + repr([round(x,7) for x in output.to_vector()]))
                print("Probability: " + repr([round(x,7) for x in output.measure()]))
        

        # Clears that "shots.txt" file, needed for multiple inputs
        abs_shots = os.path.join(os.path.dirname(__file__), "../outputFiles/shots.txt")
        f = open(abs_shots, 'w')
        f.write("")
        f.close()

        if test:
            for output in outputstates:
                temp = output.to_vector()
                if qiskit_ordering: temp = swap_significants(temp)
                return temp

        # Prints either state or output vectors
        if savestate:
            with open(abs_output, 'w') as f:
                f.write("State vectors: \n")
                for output in outputstates:
                    if shots: shoot(output.to_vector(),shots, qiskit_ordering)
                    temp = output.to_vector()
                    if qiskit_ordering: temp = swap_significants(temp)
                    f.write(repr(temp) + "\n")
        else:
            with open(abs_output, 'w') as f:
                f.write("Probabillity vectors: \n")
                for output in outputstates:
                    if shots: shoot(output.to_vector(),shots, qiskit_ordering)
                    temp = output.measure()
                    if qiskit_ordering: temp = swap_significants(temp)
                    f.write(repr([round(x,15) for x in temp]) + "\n") # The rounding is for numbers like 0.499..
        

        

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
                string += repr(qc.get_element((x, y)))
                f.write(string)
            f.write(']')
            if x != z:
                f.write(',')
        
        f.write(']')



# Collapses the vector and prints it values 'reps' amount of times. Also prints the distribution.
def shoot(vector, reps, qiskit_ordering):
    abs_shots = os.path.join(os.path.dirname(__file__), "../outputFiles/shots.txt")
    f = open(abs_shots, 'a')
    probs = [abs(x)**2 for x in vector]

    # Amount of bits
    n_bits = int(np.log2(len(vector)))
    
    # Array of all the collapsed bits
    bits = np.random.choice(
                            [np.binary_repr(x, n_bits) for x in np.arange(0,len(vector))]
                            ,reps,p=probs)

    # Swaps significans of the bits
    if qiskit_ordering:
        for index, b in enumerate(bits):
            bits[index] = b[::-1] # Flips the bit order

    # Shows statistics of how many times a value was chosen
    f.write(repr(Counter(bits)) + "\n")

    # Writes every collapsed bit
    for b in bits:
        f.write(b + "\n")



# Swaps the significant order of bits in the statevector.
# Used to represent the result the same way Qiskit does.
def swap_significants(vector):

    # Amount of bits
    n_bits = int(np.log2(len(vector)))

    swap_arr = [0]*len(vector)

    # Swaps the index of a non 0 value
    for index, b in enumerate(vector):
        if b != 0:
            rev = (np.binary_repr(index, n_bits))[::-1] # Flips the bit order
            new_index = int(rev,2) # Converts binary to decimal
            swap_arr[new_index] = b

    return swap_arr

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tree based qasm sim.')

    # Assign a path to the arg 'gates'
    parser.add_argument('-g', dest='gates', help='File with gates', required=True)
    # Assign a path to the arg 'circuit'
    parser.add_argument('-c', dest='circuit', help='qasm file with circuit', required=True)
    # Assign a path to the arg 'input_state'
    parser.add_argument('-i', dest='input_state', help='File with input state(s)')
    group = parser.add_mutually_exclusive_group()
    # Numbers of shots to return
    parser.add_argument('-shots', help='Return a number of bitstrings from final vector probabilities', type=int)
    # Prints result in the qiskit representation
    parser.add_argument('-flipOrder', action='store_true', help='Swaps the bit order to match Qiskit')
    # Wether to save the final vector or calculate probabilities
    parser.add_argument('-state', action='store_true', help='Print the final vector to output.txt instead of calculating probabilities')
    # Save the matrix instead of calculating anything with it
    group.add_argument('-saveMatrix', help='Print the final matrix to path instead of multiplying with a vector')
    # Debug: print extra info
    parser.add_argument('-debug', action='store_true', help='Print extra debug info')
    # Test: used for test file. Don't use this
    parser.add_argument('-test', action='store_true', help='Returns state vector')
    args = parser.parse_args()

    gatespath = args.gates
    circuitpath = args.circuit
    inputstate = args.input_state
    shots = args.shots
    savestate = args.state
    save_matrix = args.saveMatrix
    qiskit_ordering = args.flipOrder
    debug = args.debug
    test = -args.test

    main(gatespath, circuitpath, inputstate, shots, savestate, save_matrix, qiskit_ordering, debug, test)