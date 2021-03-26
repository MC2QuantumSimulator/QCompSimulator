import gc
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
            return False if not o else (self.conns == o.conns and self.weights == o.weights)

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

        def z_order_better(index):
            #i=0
            #for y in deBruijn:
            #    for x in deBruijn:
            #        if index == x + 2*y:
            #            return i
            #        i+=1
            #print("z_order_better sucks")
            start = 0
            sub_index = 0
            if index >= size**2 // 2:
                start += size**2//2
                if index >= size**2*3//4:
                    start+= size//2
            elif index >= size**2 // 4:
                start += size // 2

            while index != deBruijn[start%size]+2*deBruijn[start//size]:
                start +=2
                sub_index+=2
                if sub_index == size // 2:
                    start+=size // 2
                    sub_index = 0
            return [start,start+1,start+size,start+size+1]


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

        def get_parent_order(matrix_index): #Uses new root and current leg to put nodes in a list that can be used to propagate
                                #factors upwards
            q = queue.LifoQueue()
            current_leg_to_touple=(matrix_index//size,matrix_index%size)
            sub_size= 1<<(first.height-1)
            target = new_root
            while sub_size > 0:
                goto = 0
                if current_leg_to_touple[0] & sub_size:
                    goto += 2
                if current_leg_to_touple[1] & sub_size:
                    goto += 1
                q.put((target,goto))
                target = target.conns[goto]
                sub_size = sub_size >> 1
            return q

        current_leg = 0  # Only used in set_weight(). Keeps track of which bottom leg is being calculated.
        size = 1 << first.height
        deBruijn = moserDeBruijn()

        def set_weight(matrix_index):
            weight = 0
            for i in range(size):
                weight += first.get_element_no_touple((matrix_index // size) * size + i) * second.get_element_no_touple(
                    matrix_index % size + i * size)
            return weight

        if (first.height != second.height):
            raise ValueError("Dimensions do not match, mult between ", first.to_matrix(), second.to_matrix())
        q = queue.LifoQueue()
        height = first.height
        size = 2 ** height
        new_root = cls.node([None] * 4, [1] * 4)  # Will be root node of resulting tree.
        q.put((new_root, height))
        termnode = qmatrix.node(None, None)
        c1 = []
        q_prop = queue.Queue()
        global_weight = 1
        while q.qsize() != 0:
            (curr_node, height) = q.get()
            if height == 1:
                legs=z_order_better(current_leg)
                curr_node.weights = [set_weight(legs[0]), set_weight(legs[1]), set_weight(legs[2]),set_weight(legs[3])]
                parents = get_parent_order(legs[0])
                if parents.qsize() == 1:
                    curr_node.conns = [termnode] * 4
                    break
                (parent,child_index) = parents.get()
                (parent,child_index) = parents.get()
                current_leg += 4
                curr_node.conns = [termnode] * 4
                # A "sub tree" should be finished at this point. Possibly insert some cleanup here?
                if all(weight == 0 for weight in curr_node.weights):
                    curr_node = None
                    propagated_factor = 0
                else:
                    propagated_factor = next((x for x in curr_node.weights if x), None)
                    curr_node.weights = [elem / propagated_factor for elem in curr_node.weights]
                    copy = next((c1_elem for c1_elem in c1 if curr_node == c1_elem), None)
                    if copy is not None:
                        curr_node = copy
                    else:
                        c1.append(curr_node)
                    q_prop.put((parent,child_index, curr_node, propagated_factor))
                    while q_prop.qsize() != 0:
                        hej = q_prop.qsize()
                        (curr_node,child_index, child, propagated_factor) = q_prop.get()
                        #child_index = curr_node.conns.index(child)
                        curr_node.weights[child_index] *= propagated_factor
                        if propagated_factor == 0:
                            curr_node.conns[child_index] = termnode
                        # Now check if factor should be propagated upwards or not.
                        else:
                            # If prop factor is not 0, and all connections of all curr_node's children are None,
                            # a factor has not yet been propagated and this one can be. Once the factor has been propagated,
                            # this node will have a child that is not None and so no further factors will be propagated.
                            #if child_index == 0 or curr_node.conns[child_index - 1] is termnode:  # Will this crash? #This is bugged? Gotta check all child_index-n for all n available
                            if all(curr_node.conns[n] is termnode for n in range(0, child_index)):
                                curr_node.weights = [weight / propagated_factor for weight in curr_node.weights]
                                if parents.qsize() > 0:
                                    (parent,child_index) = parents.get()
                                    q_prop.put((parent,child_index, curr_node, propagated_factor))
                                else:
                                    global_weight = propagated_factor
                        copy = next((c1_elem for c1_elem in c1 if curr_node == c1_elem), None)
                        if copy is not None:
                            curr_node = copy
                        else:
                            c1.append(curr_node)
            else:
                for i in [3, 2, 1, 0]:
                    if curr_node.conns[i] is None:
                        new_node = cls.node([None] * 4, [1] * 4)
                        curr_node.conns[i] = new_node
                        q.put((new_node, height - 1))

        return qmatrix(new_root, global_weight, first.height)

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
        # list to store found unique nodes
        c1 = []
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
                normelems = [elem / nonzero for elem in elems]
                qnode = qmatrix.node([termnode]*4, normelems)
                # TODO: change to something better than O(n) (hash map eq.)
                copy = next((c1_elem for c1_elem in c1 if qnode == c1_elem), None)
                if copy is not None:
                    qnode = copy
                else:
                    c1.append(qnode)
            q1.put([qnode, nonzero])

        while q1.qsize() > 1:
            node1 = q1.get()
            node2 = q1.get()
            node3 = q1.get()
            node4 = q1.get()
            nodes = [node1[0], node2[0], node3[0], node4[0]]
            weights = [node1[1], node2[1], node3[1], node4[1]]
            if all(node is None for node in nodes):
                qbc = [None, 0]
            else:
                nonzero = next((x for x in weights if x), None)
                normelems = [weight / nonzero for weight in weights]
                qnodeinner = qmatrix.node(nodes, normelems)
                # TODO: change to something better than O(n) (hash map eq.)
                copyinner = next((c1_elem for c1_elem in c1 if qnodeinner == c1_elem), None)
                if copyinner is not None:
                    qnodeinner = copyinner
                else:
                    c1.append(qnodeinner)
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
    def copy(cls, original):
        """Returns a qmatrix that is separate from the original object"""
        # TODO: Use DFS instead to make a copy of a qmatrix. Prototype version; to_matrix -> to_tree
        return qmatrix.to_tree(original.to_matrix())
