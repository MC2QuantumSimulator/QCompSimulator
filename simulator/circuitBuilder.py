from qmatrix import qmatrix

def build_circuit(gate) -> qmatrix:
	"""Constructs a matrix representing a quantum circuit"""
	return qmatrix.to_tree(gate)