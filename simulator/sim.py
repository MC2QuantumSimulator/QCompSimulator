import argparse
import ParseInput
import circuitBuilder

def main():
    parser = argparse.ArgumentParser(description='Tree based qasm simulator.')

    # Assign a path to the arg 'gates'
    parser.add_argument('-g', dest='gates', help='File with gates', required=True)
    # Assign a path to the arg 'circuit'
    parser.add_argument('-c', dest='circuit', help='qasm file with circuit', required=True)
    # Assign a path to the arg 'input_state'
    parser.add_argument('-i', dest='input_state', help='File with input state(s)')
    # Default number of repetitions
    defaultint = 100
    parser.add_argument('--shots', dest='num_reps', help='Number of repetitions. Default = {}'.format(defaultint), type=int, default=defaultint) # Ask Miroslav about a good default
    # Wether to save the final vector or calculate probabilities
    parser.add_argument('--state', action="store_true", help='Save the final vector before calculating probabilities')

    args = parser.parse_args()

    gatespath = args.gates
    circuitpath = args.circuit
    inputstate = args.input_state
    numreps = args.num_reps
    savestate = args.state

    # Print out the input to see that it worked
    print('gatepath:    ' + gatespath)
    print('circuitpath: ' + circuitpath)
    print('input state: ' + ('No input' if not inputstate else inputstate))
    print('num of reps: {}'.format(numreps))
    print('Save state:  {}'.format(savestate))

    # pass the gates path to ParseInput ant print the returned lists
    matnames, matlist = ParseInput.ParseInput.parse_gates(gatespath)
    for i, mat in enumerate(matlist):
        print(matnames[i])
        print(mat)

    # Parse the qasm file before sending to circuitbuilder?
    # Returns the final matrix? TODO: WHAT ELSE COULD IT RETURN?
    # Should circuitBuilder be a class that also includes measurment funcs and such or should that be in a different place?
    qc = circuitBuilder.build_circuit()


if __name__ == '__main__':
    main()