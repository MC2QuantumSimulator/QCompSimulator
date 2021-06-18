import math
import queue
import sys
import numpy as np
from functools import lru_cache

class qmatrix():

    # Recursion limit can be changed
    sys.setrecursionlimit(1500)
    cache_size = 1024

    class node():

        def __init__(self, conns: list, weights: list):
            self.conns = conns
            self.weights = weights

        def __hash__(self) -> int:
            return hash((0 if not self.conns else (0 if not conn else id(conn) for conn in self.conns), self.weights))

        def __eq__(self, o: object) -> bool:
            """Assumes only one copy of earlier nodes exist"""
            if not o:
                return False
            if not self.conns and not o.conns:
                return self.weights == o.weights
            return self.weights == o.weights and all(id(x) == id(y) for x,y in zip(self.conns, o.conns))

    def __init__(self, root: node, weight: complex = 1.0, height: int = 1, termination = None):
        self.root = root
        self.weight = weight
        self.height = height
        self.termination = termination

    @classmethod
    @lru_cache(maxsize=cache_size)
    def cache_node(cls, conns, weights):
        return cls.node(conns, weights)

    @classmethod
    @lru_cache(maxsize=cache_size)
    def add_nodes(cls, first: node, second: node, height: int, weights_parent: tuple, termnode) -> node:
        if not first and not second:
            return (None, 0)
        if not first:
            firstconns = (None, None, None, None)
            firstweights = (0, 0, 0, 0)
        else:
            firstconns = first.conns
            firstweights = first.weights
        if not second:
            secondconns = (None, None, None, None)
            secondweights = (0, 0, 0, 0)
        else:
            secondconns = second.conns
            secondweights = second.weights
        if height <= 1:
            weights_here1 = tuple([x*weights_parent[0] for x in firstweights])
            weights_here2 = tuple([x*weights_parent[1] for x in secondweights])
            weights_here = tuple([sum(x) for x in zip(weights_here1, weights_here2)])
            nonzero = next((x for x in weights_here if x), 1)
            normelems = tuple([weight / nonzero for weight in weights_here])
            node = cls.cache_node((termnode, termnode, termnode, termnode), normelems)
            return (node, nonzero)
        conns_n_weights = tuple([cls.add_nodes(x, y, height-1, (z*weights_parent[0], w*weights_parent[1]), termnode) for x,y,z,w in zip(firstconns, secondconns, firstweights, secondweights)])
        conns = tuple([None if not item else item[0] for item in conns_n_weights])
        weights_from_children = tuple([0 if not item else item[1] for item in conns_n_weights])
        nonzero = next((x for x in weights_from_children if x), 1)
        normelems = tuple([weight / nonzero for weight in weights_from_children])
        node = cls.cache_node(conns, normelems)
        return (node, nonzero)

    @classmethod
    def add_matrices(cls, first, second):
        # Used for debugging the add_nodes func, not needed for sim
        termnode = cls.node(None, None)
        new_node, norm = cls.add_nodes(first.root, second.root, first.height, (first.weight,second.weight), termnode)
        return cls(new_node, norm, first.height, first.termination)

    @classmethod
    @lru_cache(maxsize=cache_size)
    def mult_nodes(cls, first: node, second: node, height: int, weight_from_parent: float, termnode: node) -> node:
        if not first or not second:
            return (None, 0)
        if height <= 1:
            newweightsleft = tuple([first.weights[x]*second.weights[y] for x,y in zip((0,0,2,2),(0,1,0,1))])
            newweightsright = tuple([first.weights[x]*second.weights[y] for x,y in zip((1,1,3,3),(2,3,2,3))])
            retweights = tuple([x+y for x,y in zip(newweightsleft, newweightsright)])
            nonzero = next((x for x in retweights if x), 1)
            normelems = tuple([weight / nonzero for weight in retweights])
            node = cls.cache_node((termnode, termnode, termnode, termnode), normelems)
            return (node, nonzero*weight_from_parent)
        newconnsleft_n_weights = tuple([cls.mult_nodes(first.conns[x], second.conns[y], height-1, first.weights[x]*second.weights[y], termnode) for x,y in zip((0,0,2,2),(0,1,0,1))])
        newconnsright_n_weights = tuple([cls.mult_nodes(first.conns[x], second.conns[y], height-1, first.weights[x]*second.weights[y], termnode) for x,y in zip((1,1,3,3),(2,3,2,3))])
        newconnsleft = tuple([None if not item else item[0] for item in newconnsleft_n_weights])
        newconnsright = tuple([None if not item else item[0] for item in newconnsright_n_weights])
        weightpropleft = tuple([0 if not item else item[1] for item in newconnsleft_n_weights])
        weightpropright = tuple([0 if not item else item[1] for item in newconnsright_n_weights])
        newleft = cls.node(newconnsleft, weightpropleft)
        newright = cls.node(newconnsright, weightpropright)
        result, weight = cls.add_nodes(newleft, newright, height, (1, 1), termnode)
        return (result, weight*weight_from_parent)

    @classmethod
    def mult(cls, first, second):
        termnode = cls.node(None, None)
        new_node, weight = cls.mult_nodes(first.root, second.root, first.height, first.weight*second.weight, termnode)
        return cls(new_node, weight, first.height, termnode)

    def get_element(self, index: tuple) -> complex:
        size = 1 << (self.height-1)
        #if (element >= size<<1 or element < 0):
        #	raise ValueError("Element out of bounds, element was {} when allowed values are 0 - {}".format(element, size-1))
        value = self.weight
        target = self.root
        while size > 0:
            goto = 0
            if index[0] & size:
                goto += 2
            if index[1] & size:
                goto += 1
            if target.weights[goto] == 0:
                    return 0
            value *= target.weights[goto]
            target = target.conns[goto]
            size = size >> 1

        return value

    def to_matrix(self):
        size = 1 << (self.height)
        arr = []
        for i in range(size):
            locarr = []
            for j in range(size):
                locarr.append(self.get_element((i,j)))
            arr.append(locarr)
        return np.array(arr)

    @staticmethod
    def get_matrix_element(matrix: np.ndarray, index: int) -> complex:
        size = matrix.size >> 1 # is 2^2n-1, only one bit is 1
        offset = matrix.shape[0] >> 1
        x = 0
        y = 0
        while size > 0:
            if index & size:
                y += offset
            size = size >> 1
            if index & size:
                x += offset
            size = size >> 1
            offset = offset >> 1

        return matrix.item((y, x))

    @staticmethod
    def to_tree(matrix:np.ndarray): # TODO: break out parts of the function to reduce Cognitive Complexity + reuse parts
        # possible changes: change from queue to array. This allows for parallelization better.
        "Returns a qmatrix tree from a matrix"
        q1 = queue.Queue()
        # Create a set to store found unique nodes
        c1 = set()
        shape = matrix.shape
        if matrix.ndim != 2:
            raise ValueError("Number of array dimensions was not 2, was {}".format(matrix.ndim))
        if shape[0] != shape[1]:
            raise ValueError("Array size was not equal for both directions, was ({},{})".format(shape[0], shape[1]))
        n = shape[0]
        if (n & (n-1) != 0) or n < 2:
            raise ValueError("Matrix size is not a power of two, size is {} by {}".format(n, n))

        height = int(math.log2(n))
        termnode = qmatrix.node(None, None)
        for i in range(matrix.size >> 2):
            elems = []
            for j in range(4):
                elems.append(qmatrix.get_matrix_element(matrix, 4*i+j))
            if all(elem == 0 for elem in elems):
                qnode = None
                nonzero = 0
            else:
                nonzero = next((x for x in elems if x), None)
                norm_elems = tuple([elem / nonzero for elem in elems])
                qnode = qmatrix.node((termnode, termnode, termnode, termnode), norm_elems)
                copy = next((c1_elem for c1_elem in c1 if qnode == c1_elem), None)
                if copy is not None:
                    qnode = copy
                else:
                    c1.add(qnode)
            q1.put([qnode, nonzero])

        qmatrix.merge_tree(q1, c1)
        (root, weight) = q1.get()
        return qmatrix(root, weight, height, termnode)

    @staticmethod
    def merge_tree(q1, c1):
        # Loop until only one node, the root, remains
        while q1.qsize() > 1:
            # Get 4 elements from the queue and palce in lists
            node1 = q1.get()
            node2 = q1.get()
            node3 = q1.get()
            node4 = q1.get()
            nodes = (node1[0], node2[0], node3[0], node4[0])
            weights = [node1[1], node2[1], node3[1], node4[1]]
            # Return None if all subtrees are None
            if all(node is None for node in nodes):
                qbc = [None, 0]
            else:
                # Get first non zero element
                nonzero = next((x for x in weights if x), None)
                # Normalize with that element, always results in the first non zero element to be 1
                norm_elems = tuple([weight / nonzero for weight in weights])
                qnode_inner = qmatrix.node(nodes, norm_elems)
                # Check if an identical node exists and replace with that one if so
                copy_inner = next((c1_elem for c1_elem in c1 if qnode_inner == c1_elem), None)
                if copy_inner is not None:
                    qnode_inner = copy_inner
                else:
                    c1.add(qnode_inner)
                qbc = [qnode_inner, nonzero]
            # Put new node back in the queue
            q1.put(qbc)

    @classmethod
    def kron(cls, first, target):
        "Returns the kronecker product of first and target, consuming the trees in the process"
        # kron on itself would cause an infinite loop in the tree, raise error if attempted
        if first is target:
            raise ValueError("Can not perform Kronecker product on itself, objects are the same")

        # add the data inside target.root to first.termination and then create new qmatrix
        first.termination.conns = target.root.conns
        first.termination.weights = target.root.weights
        result = cls(first.root, first.weight*target.weight, first.height+target.height, target.termination)

        # Nuke first and target
        first.root = None
        first.termination = None
        target.root = None
        target.termination = None
        return result

    @classmethod
    def id(cls, n):
        "Returns an identity matrix of size 2^n, equivalent to n qubits."
        # Create the first layer
        orig_term = cls.node(None, None)
        identity = cls(cls.node((orig_term,None,None,orig_term), (1,0,0,1)), termination=orig_term)
        # Add n-1 more layers, pointing at the new result each time
        for _ in range(n-1):
            new_term = cls.node(None, None)
            new_node = cls(cls.node((new_term,None,None,new_term), (1,0,0,1)), termination=new_term)
            identity = cls.kron(identity, new_node)
        return identity

    def number_of_nodes(self):
        "Returns the number of unique nodes in the tree, excluding the termination node"
        # LIFO queue to store nodes to be traversed
        s1 = queue.LifoQueue()
        # set storing all found nodes
        c1 = set()

        # add root node to queue and the set
        s1.put((self.root, self.height))
        c1.add(self.root)
        sum_nodes =1

        while s1.qsize() != 0:
            curr, height = s1.get()
            if curr and (height > 1):

                # if the current conns have not already been found; add them to the queue and the set
                for conn in curr.conns:
                    if conn and conn not in c1:
                        s1.put((conn, height-1))
                        c1.add(conn)
                        sum_nodes += 1

        return sum_nodes