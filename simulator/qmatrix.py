import queue
import numpy as np

class qmatrix():
	
	class node():

		def __init__(self, conns: list, weights: list, depth: int = 0):
			self.depth = depth
			self.conns = conns
			self.weights = weights

		@classmethod
		def merge(cls, nodes): # Add propagation of factors.
			"""Merges four nodes into a single node of depth one larger"""
			if all(node is None for node in nodes):
				raise ValueError("All nodes to be merged are 'None', at most three 'None' allowed") # should we allow all None and return None here?
			#if (all(node is not None for node in nodes) and (node1.depth != node2.depth)): # TODO: all not None should be equal
			#	raise ValueError("Depth is not equal on nodes to be merged, {} != {}".format(node1.depth, node2.depth))
			return cls(nodes, [1 if node is not None else 0 for node in nodes], max([0 if node is None else node.depth for node in nodes]) + 1)

	def __init__(self, root: node, weight: complex = 1.0):
		self.root = root
		self.weight = weight

	def get_element(self, element: tuple) -> complex:
		size = 1<<self.root.depth
		#if (element >= size<<1 or element < 0):
		#	raise ValueError("Element out of bounds, element was {} when allowed values are 0 - {}".format(element, size-1))
		value = self.weight
		target = self.root
		while size > 0:
			goto = 0
			if element[0]&size:
				goto += 2
			if element[1]&size:
				goto += 1
			if target.weights[goto] == 0:
					return 0
			value *= target.weights[goto]
			target = target.conns[goto]
			size = size>>1

		return value

	def to_matrix(self):
		size = 1<<(self.root.depth+1)
		arr = []
		for i in range(size):
			locarr = []
			for j in range(size):
				locarr.append(self.get_element((i,j)))
			arr.append(locarr)
		return np.array(arr)
	
	@staticmethod
	def get_matrix_element(matrix: np.ndarray, element: int) -> complex:
		size = matrix.size>>1 # is 2^2n-1, only one bit is 1
		offset = matrix.shape[0]>>1
		x = 0
		y = 0
		while size > 0:
			if element&size:
				y += offset
			size = size>>1
			if element&size:
				x += offset
			size = size>>1
			offset = offset>>1

		return matrix.item((y, x))

	@staticmethod
	def to_tree(matrix:np.ndarray): # Does NOT propagate GCD values up, does remove zero nodes in an ugly way. Adding propagation of factors can be done
		# possible changes: change from queue to array. This allows for parallelization better.
		"""Returns a qmatrix tree from a matrix"""
		q1 = queue.Queue()
		shape = matrix.shape
		if matrix.ndim != 2:
			raise ValueError("Array dimensions was not 2, was {}".format(matrix.ndim))
		if shape[0] != shape[1]:
			raise ValueError("Array size was not square, was ({},{})".format(shape[0], shape[1]))
		n = shape[0]
		if (n & (n-1) != 0) or n < 2:
			raise ValueError("Matrix size is not a power of two, size is {}".format(n))
		for i in range(matrix.size>>2):
			elems = []
			for j in range(4):
				elems.append(qmatrix.get_matrix_element(matrix, 4*i+j))
			if all(elem == 0 for elem in elems):
				qmat = None
			else:
				qmat = qmatrix.node([None]*4, elems)
			q1.put(qmat)

		while q1.qsize() > 1:
			node1 = q1.get()
			node2 = q1.get()
			node3 = q1.get()
			node4 = q1.get()
			nodes = (node1, node2, node3, node4)
			if all(node is None for node in nodes):
				qbc = None
			else:
				qbc = qmatrix.node.merge(nodes)
			q1.put(qbc)
		root = q1.get()
		return qmatrix(root)
