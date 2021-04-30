from numpy import pi
from qiskit.circuit.quantumcircuit import QuantumCircuit 

def qft_rotations(circuit, n):
    """Performs qft on the first n qubits in circuit (without swaps)"""
    if n == 0:
        return circuit
    n -= 1
    circuit.h(n)
    for qubit in range(n):
        circuit.cp(pi/2**(n-qubit), qubit, n)
    # At the end of our function, we call the same function again on
    # the next qubits (we reduced n by one earlier in the function)
    qft_rotations(circuit, n)

def swap_registers(circuit, n):
    for qubit in range(n//2):
        circuit.swap(qubit, n-qubit-1)
    return circuit

def qft(circuit, n):
    """QFT on the first n qubits in circuit"""
    qft_rotations(circuit, n)
    swap_registers(circuit, n)
    return circuit

def create_qft(n):
    qc = QuantumCircuit(n)
    qft(qc,n)
    return qc.qasm()

if __name__ == "__main__":
    n = 4
    text = create_qft(n)
    with open("inputFiles/testQASMprint.txt", "w") as f:
        f.write(text)