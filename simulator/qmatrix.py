import math
import queue
import numpy as np
from functools import lru_cache

class qmatrix():

    # Recursion limit can be changed
    cache_size = None

    class node():

        def __init__(self, conns: list, weights: list):
            self.conns = conns
            self.weights = weights

        # def __hash__(self) -> int:
        #     return hash((self.conns, self.weights))

        # def __eq__(self, o: object) -> bool:
        #     """Assumes only one copy of earlier nodes exist"""
        #     return False if not o else (self.conns == o.conns and self.weights == o.weights)

        @classmethod
        def merge(cls, nodes, heights): # Add propagation of factors.
            """Merges four nodes into a single node of height one larger"""
            if all(node is None for node in nodes):
                raise ValueError("All nodes to be merged are 'None', at most three 'None' allowed") # should we allow all None and return None here?
            #if (all(node is not None for node in nodes) and (node1.height != node2.height)): # TODO: all not None should be equal
            #	raise ValueError("height is not equal on nodes to be merged, {} != {}".format(node1.height, node2.height))
            return cls(nodes, [1 if node is not None else 0 for node in nodes]), max([1 if height is None else height for height in heights]) + 1

    def __init__(self, root: node, weight: complex = 1.0, height: int = 1, termination = None):
        self.root = root
        self.weight = weight
        self.height = height
        self.termination = termination

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
        if height > 1:
            conns_n_weights = tuple([cls.add_nodes(x, y, height-1, (z*weights_parent[0], w*weights_parent[1]), termnode) for x,y,z,w in zip(firstconns, secondconns, firstweights, secondweights)])
            conns = tuple([None if not item else item[0] for item in conns_n_weights])
            weights_from_children = tuple([0 if not item else item[1] for item in conns_n_weights])
        else:
            conns = (None, None)
        if height <= 1:
            weights_here1 = tuple([x*weights_parent[0] for x in firstweights])
            weights_here2 = tuple([x*weights_parent[1] for x in secondweights])
            weights_here = tuple([sum(x) for x in zip(weights_here1, weights_here2)])
            nonzero = next((x for x in weights_here if x), 1)
            normelems = tuple([weight / nonzero for weight in weights_here])
            node = cls.node((termnode, termnode, termnode, termnode), normelems)
            return (node, nonzero)

        nonzero = next((x for x in weights_from_children if x), None)
        if not nonzero:
            nonzero = 1
        normelems = tuple([weight / nonzero for weight in weights_from_children])
        
        node = cls.node(conns, normelems)
        return (node, nonzero)

    @classmethod
    def add_matrices(cls, first, second):
        # TODO: nuke first and second
        # Used for debugging the add_nodes func, not needed for sim
        termnode = first.termination
        new_node, norm = cls.add_nodes(first.root, second.root, first.height, (first.weight,second.weight), termnode)
        return cls(new_node, norm, first.height, first.termination)

    @classmethod
    @lru_cache(maxsize=cache_size)
    def mult_nodes(cls, first: node, second: node, height: int, termnode: node) -> node:
        if not first or not second:
            return (None, 0)
        newweightsleft = tuple([first.weights[x]*second.weights[y] for x,y in zip((0,0,2,2),(0,1,0,1))])
        newweightsright = tuple([first.weights[x]*second.weights[y] for x,y in zip((1,1,3,3),(2,3,2,3))])
        if height <= 1:
            retweights = tuple([x+y for x,y in zip(newweightsleft, newweightsright)])
            nonzero = next((x for x in retweights if x), 1)
            normelems = tuple([weight / nonzero for weight in retweights])
            node = cls.node((termnode, termnode, termnode, termnode), normelems)
            return (node, nonzero)
        newconnsleft_n_weights = tuple([cls.mult_nodes(first.conns[x], second.conns[y], height-1, termnode) for x,y in zip((0,0,2,2),(0,1,0,1))])
        newconnsright_n_weights = tuple([cls.mult_nodes(first.conns[x], second.conns[y], height-1, termnode) for x,y in zip((1,1,3,3),(2,3,2,3))])
        newconnsleft = tuple([None if not item else item[0] for item in newconnsleft_n_weights])
        newconnsright = tuple([None if not item else item[0] for item in newconnsright_n_weights])
        weightpropleft = tuple([0 if not item else item[1] for item in newconnsleft_n_weights])
        weightpropright = tuple([0 if not item else item[1] for item in newconnsright_n_weights])
        resweightsleft = tuple([x*y for x,y in zip(newweightsleft, weightpropleft)])
        resweightsright = tuple([x*y for x,y in zip(newweightsright, weightpropright)])
        newleft = cls.node(newconnsleft, resweightsleft)
        newright = cls.node(newconnsright, resweightsright)
        result, weight = cls.add_nodes(newleft, newright, height, (1,1), termnode)
        return (result, weight)

    @classmethod
    def mult(cls, first, second):
        # TODO: nuke first and second
        termnode = first.termination
        new_node, weight = cls.mult_nodes(first.root, second.root, first.height, termnode)
        return cls(new_node, weight, first.height, termnode)

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

    def get_element_no_touple(self,element): #USED IN MATRIX X VECTOR MULT IN QVECTOR
        size = 1<<(self.height)
        element_to_touple=(element//size,element%size)
        return self.get_element(element_to_touple)

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
    def to_tree(matrix:np.ndarray): # TODO: break out parts of the function to reduce Cognitive Complexity + reuse parts
        # possible changes: change from queue to array. This allows for parallelization better.
        """Returns a qmatrix tree from a matrix"""
        q1 = queue.Queue()
        # set to store found unique nodes
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
        for i in range(matrix.size>>2):
            elems = []
            for j in range(4):
                elems.append(qmatrix.get_matrix_element(matrix, 4*i+j))
            if all(elem == 0 for elem in elems):
                qnode = None
                nonzero = 0
            else:
                nonzero = next((x for x in elems if x), None)
                normelems = tuple([elem / nonzero for elem in elems])
                qnode = qmatrix.node((termnode, termnode, termnode, termnode), normelems)
                copy = next((c1_elem for c1_elem in c1 if qnode == c1_elem), None)
                if copy is not None:
                    qnode = copy
                else:
                    c1.add(qnode)
            q1.put([qnode, nonzero])

        while q1.qsize() > 1:
            node1 = q1.get()
            node2 = q1.get()
            node3 = q1.get()
            node4 = q1.get()
            nodes = (node1[0], node2[0], node3[0], node4[0])
            weights = [node1[1], node2[1], node3[1], node4[1]]
            if all(node is None for node in nodes):
                qbc = [None, 0]
            else:
                nonzero = next((x for x in weights if x), None)
                normelems = tuple([weight / nonzero for weight in weights])
                qnodeinner = qmatrix.node(nodes, normelems)
                copyinner = next((c1_elem for c1_elem in c1 if qnodeinner == c1_elem), None)
                if copyinner is not None:
                    qnodeinner = copyinner
                else:
                    c1.add(qnodeinner)
                qbc = [qnodeinner, nonzero]
            q1.put(qbc)
        (root, weight) = q1.get()
        return qmatrix(root, weight, height, termnode)

    @classmethod
    def kron(cls, first, target):
        """Returns the kronecker product of first and target, consuming the trees in the process"""
        # kron on itself would cause an infinite loop in the tree, raise error if attempted
        if first is target:
            raise ValueError("Can not perform Kronecker product on itself, objects are the same")

        # add the data inside target.root to first.termination and then create new qmatrix
        first.termination.conns = target.root.conns
        first.termination.weights = target.root.weights
        result = qmatrix(first.root, first.weight*target.weight, first.height+target.height, target.termination)

        # Nuke first and target
        first.root = None
        first.termination = None
        target.root = None
        target.termination = None
        return result

    @classmethod
    def id(cls, n):
        """Returns an identity matrix of size 2^n, equivalent to n qubits."""
        # Create the first layer
        origterm = cls.node(None, None)
        identity = cls(cls.node((origterm,None,None,origterm), (1,0,0,1)), termination=origterm)
        # Add n-1 more layers, pointing at the new result each time
        for _ in range(n-1):
            newterm = cls.node(None, None)
            newnode = cls(cls.node((newterm,None,None,newterm), (1,0,0,1)), termination=newterm)
            identity = cls.kron(identity, newnode)
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

    @classmethod
    def copy(cls, original):
        """Returns a qmatrix that is separate from the original object"""
        # TODO: Use DFS instead to make a copy of a qmatrix. Prototype version; to_matrix -> to_tree
        return qmatrix.to_tree(original.to_matrix())
