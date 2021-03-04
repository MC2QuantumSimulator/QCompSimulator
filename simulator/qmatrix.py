import queue
import numpy as np

class qmatrix():
    
    class node():

        def __init__(self, conns: list, weights: list):
            self.conns = conns
            self.weights = weights

        @classmethod
        def merge(cls, nodes, heights): # Add propagation of factors.
            """Merges four nodes into a single node of height one larger"""
            if all(node is None for node in nodes):
                raise ValueError("All nodes to be merged are 'None', at most three 'None' allowed") # should we allow all None and return None here?
            #if (all(node is not None for node in nodes) and (node1.height != node2.height)): # TODO: all not None should be equal
            #	raise ValueError("height is not equal on nodes to be merged, {} != {}".format(node1.height, node2.height))
            return cls(nodes, [1 if node is not None else 0 for node in nodes]), max([1 if height is None else height for height in heights]) + 1

    def __init__(self, root: node, weight: complex = 1.0, height: int = 1):
        self.root = root
        self.weight = weight
        self.height = height

    def get_element(self, index: tuple) -> complex:
        size = 1<<(self.height-1)
        #if (element >= size<<1 or element < 0):
        #	raise ValueError("Element out of bounds, element was {} when allowed values are 0 - {}".format(element, size-1))
        value = self.weight
        target = self.root
        while size > 0:
            goto = 0
            if index[0]&size:
                goto += 2
            if index[1]&size:
                goto += 1
            if target.weights[goto] == 0:
                    return 0
            value *= target.weights[goto]
            target = target.conns[goto]
            size = size>>1

        return value

    def to_matrix(self):
        size = 1<<(self.height)
        arr = []
        for i in range(size):
            locarr = []
            for j in range(size):
                locarr.append(self.get_element((i,j)))
            arr.append(locarr)
        return np.array(arr)
    
    @staticmethod
    def get_matrix_element(matrix: np.ndarray, index: int) -> complex:
        size = matrix.size>>1 # is 2^2n-1, only one bit is 1
        offset = matrix.shape[0]>>1
        x = 0
        y = 0
        while size > 0:
            if index&size:
                y += offset
            size = size>>1
            if index&size:
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
            raise ValueError("Number of array dimensions was not 2, was {}".format(matrix.ndim))
        if shape[0] != shape[1]:
            raise ValueError("Array size was not equal for both directions, was ({},{})".format(shape[0], shape[1]))
        n = shape[0]
        if (n & (n-1) != 0) or n < 2:
            raise ValueError("Matrix size is not a power of two, size is {} by {}".format(n, n))
        for i in range(matrix.size>>2):
            elems = []
            for j in range(4):
                elems.append(qmatrix.get_matrix_element(matrix, 4*i+j))
            if all(elem == 0 for elem in elems):
                qmat = None
            else:
                qmat = qmatrix.node([None]*4, elems)
            q1.put((qmat, 1))

        while q1.qsize() > 1:
            node1 = q1.get()
            node2 = q1.get()
            node3 = q1.get()
            node4 = q1.get()
            nodes = (node1[0], node2[0], node3[0], node4[0])
            heights = (node1[1], node2[1], node3[1], node4[1])
            if all(node is None for node in nodes):
                qbc = None
            else:
                qbc = qmatrix.node.merge(nodes, heights) # tuple, node & height
            q1.put(qbc)
        (root, height) = q1.get()
        return qmatrix(root, 1, height)

    @classmethod
    def kron(cls, first, target): # REUSES FIRST!!!
        if first is target:
            raise ValueError("Can not perform Kronecker product on itself, obects are the same")
        # Initializing a stack of for all nodes
        s1 = queue.LifoQueue()

        # attach rootnode to stack
        s1.put((first.root, first.height))

        while s1.qsize() != 0:
            curr, height = s1.get()
            if height > 1:
                for conn in curr.conns:
                    if conn:
                        s1.put((conn, height-1))

            if all(conn is None for conn in curr.conns) and height <= 1:
                curr.conns = [target.root]*4
        first.height += target.height
        first.weight *= target.weight

    @classmethod
    def copy(cls, original):
        """Returns a qmatrix that is separate from the original object"""
        # TODO: Use DFS instead to make a copy of a qmatrix. Prototype version; to_matrix -> to_tree
        return qmatrix.to_tree(original.to_matrix())