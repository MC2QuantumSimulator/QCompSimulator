import argparse

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

    print('gatepath: ' + gatespath)
    print('circuitpath: ' + circuitpath)


if __name__ == '__main__':
    main()