import math
import queue
import numpy as np

class qmatrix():
    
    class node():

        def __init__(self, conns: list, weights: list):
            self.conns = conns
            self.weights = weights

        def __hash__(self) -> int:
            return hash((self.conns, self.weights))

        def __eq__(self, o: object) -> bool:
            """Assumes only one copy of earlier nodes exist"""
            return self.conns == o.conns and self.weights == o.weights

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
    def mult(cls, first, second):
        # Plan: Create node from the top with no childsm and put it in a queue. Take node from queue and check if it's childless. If it is, create childs of lower height and put in queue. Get from queue ( last in first out ) same node and check if childless,
        # if yes, create child. When at depth 1, set weights. Doing it this way should finish one side of the tree first. When weights have been set, can start propagating factors.
        def moserDeBruijn():  # Will be a vector of length 2^qubits=size
            def gen(n):
                S = [0, 1]
                for i in range(2, n + 1):
                    if i % 2 == 0:
                        S.append(4 * S[int(i / 2)])
                    else:
                        S.append(4 * S[int(i / 2)] + 1)
                z = S[n]
                return z

            sequence = []
            for i in range(size):
                sequence.append(gen(i))
            return sequence

        def z_order_indexing(deBruijn, index):
            y = [elem * 2 for elem in deBruijn]
            return (y[index // size] + deBruijn[index % size])

        current_leg = 0  # Only used in set_weight(). Keeps track of which bottom leg is being calculated.
        size =1 << first.height
        deBruijn = moserDeBruijn()

        def set_weight(current_leg):
            for i in range(size**2):
                if current_leg == z_order_indexing(deBruijn,i):
                    matrix_index=i
                    break
            weight = 0
            for i in range(size):
                weight += first.get_element_no_touple((matrix_index // size)*size + i) * second.get_element_no_touple(
                    matrix_index % size + i * size)
            return weight

        if (first.height != second.height):
            raise ValueError("Dimensions do not match, mult between ", first.to_matrix(), second.to_matrix())
        q = queue.LifoQueue()
        height = first.height
        size = 2 ** height
        new_root = cls.node([None] * 4, [1]*4)  # Will be root node of resulting tree.
        q.put((new_root, height))
        while q.qsize() != 0:
            (curr_node, height) = q.get()
            if height == 1:
                curr_node.weights = (set_weight(current_leg), set_weight(current_leg+1), set_weight(current_leg+2), set_weight(current_leg+3))
                current_leg+=4
                # A "sub tree" should be finished at this point. Possibly insert some cleanup here?
            else:
                for i in [3, 2, 1, 0]:
                    if curr_node.conns[i] is None:
                        new_node = cls.node([None] * 4, [1]*4)
                        curr_node.conns[i] = new_node
                        q.put((new_node, height - 1))
        return qmatrix(new_root, 1, first.height)

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

    @classmethod
    def copy(cls, original):
        """Returns a qmatrix that is separate from the original object"""
        # TODO: Use DFS instead to make a copy of a qmatrix. Prototype version; to_matrix -> to_tree
        return qmatrix.to_tree(original.to_matrix())
