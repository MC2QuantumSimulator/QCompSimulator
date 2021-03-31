import argparse
import numpy as np
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
    group.add_argument('--shots', help='Return a number of bitstrings from final vector probabilities', type=int)
    # Wether to save the final vector or calculate probabilities
    group.add_argument('--state', help='Print the final vector to path instead of calculating probabilities')
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

    if debug:
        # Print out the input to see that it worked
        print('gatepath:    ' + gatespath)
        print('circuitpath: ' + circuitpath)
        print('input state: ' + ('No input, defaulting to |0>' if not inputstate else inputstate))
        if shots:
            print('num of reps: {}'.format(shots))
        print('Save state:  {}'.format(savestate))

    # pass the gates path to ParseInput ant print the returned lists
    #matnames, matlist = ParseInput.ParseInput.parse_gates(gatespath)
    #for i, mat in enumerate(matlist):
    #    print(matnames[i])
    #    print(mat)

    gate_names, gate_matrix = ParseInput.ParseInput.parse_gates(gatespath)

    qmats, height = interpreter.parse_qasm(circuitpath, gate_names, gate_matrix)

    q = interpreter.qreg(height)

    for qmat in qmats:
        q = qvector.mult(qmat, q)
    print(q.to_vector())
    print(q.measure())

    


    # Returns a matrix tree when passed a path to the qasm file and the parsed gates (or call parse gates from this func?)
    # Should circuitBuilder be a class that also includes measurment funcs and such or should that be in a different place?
    # TODO: Repalce with actual qasm parser & circuit builder
    #qc = circuitBuilder.build_circuit(np.array(matlist[0]))

    # Prints the resulting matrix in a Python/Matlab compatible format
    # TODO: Make it possible to output mathematical 'stuff' (exp, roots)
    # TODO: parse input states, currently no input state can be parsed, only default works
    #if save_matrix:
    #    write_matrix_to_file(save_matrix, qc)
    #else:
    #    if not inputstate:
    #        input_vector = np.zeros(1<<height)
    #        input_vector[0] = 1
    #    input_tree = qvector.to_tree(input_vector)
    #    qv = qvector.mult(qc, input_tree)
#
    #    if savestate:
    #        with open(savestate, 'w') as f:
    #            f.write(str(qv.to_vector()))

def write_matrix_to_file(save_matrix, qc):
    with open(save_matrix, 'w') as f:
        f.write('[')
        for x in range(1 << qc.height):
            if x != 0:
                f.write('\n')
            f.write('[')
            for y in range(1 << qc.height):
                string = ', ' if y != 0 else ''
                string += str(qc.get_element((x, y)))
                f.write(string)
            f.write('],')
        f.write(']')


if __name__ == '__main__':
    main()