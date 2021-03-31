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
    def add_nodes(cls, first: node, second: node, height: int, weights_parent: tuple):
        if not first:
            return second
        if not second:
            return first
        weight_parent_sum = sum(weights_parent)
        weights_here1 = tuple([x*weights_parent[0] for x in first.weights])
        #weights_here1 = tuple([x*y for x,y in zip(first.weights,weights_parent)])
        weights_here2 = tuple([x*weights_parent[1] for x in second.weights])
        #weights_here2 = tuple([x*y for x,y in zip(second.weights,weights_parent)])
        weights_here = tuple([sum(x) for x in zip(weights_here1, weights_here2)])
        weight_div = tuple([x/weight_parent_sum for x in weights_here])
        if height <= 1:
            return cls.node((None, None, None, None), weight_div)
        conns = [cls.add_nodes(x, y, height-1, (z*weights_parent[0], w*weights_parent[1])) for x,y,z,w in zip(first.conns, second.conns, first.weights, second.weights)]
        return cls.node(conns, weight_div)

    @classmethod
    def add_matrices(cls, first, second):
        new_node = cls.add_nodes(first.root, second.root, first.height, (first.weight,second.weight))
        return cls(new_node, first.weight+second.weight, first.height, first.termination)

    @classmethod
    def mult(cls, first, second):
        # Plan: Create node from the top with no childsm and put it in a queue. Take node from queue and check if it's childless. If it is, create childs of lower height and put in queue. Get from queue ( last in first out ) same node and check if childless,
        # if yes, create child. When at depth 1, set weights. Doing it this way should finish one side of the tree first. When weights have been set, can start propagating factors.

        def z_order_better(index, size, deBruijn):
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

        def moserDeBruijn(size):  # Will be a vector of length 2^qubits=size
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

        def get_parent_order(matrix_index, first): #Uses new root and current leg to put nodes in a list that can be used to propagate
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

        
        def set_weight(matrix_index, size):
            weight = 0
            for i in range(size):
                weight += first.get_element_no_touple((matrix_index // size) * size + i) * second.get_element_no_touple(
                    matrix_index % size + i * size)
            return weight

        if (first.height != second.height):
            raise ValueError("Dimensions do not match, mult between ", first.to_matrix(), second.to_matrix())

        height = first.height
        size = 2 ** height
        deBruijn = moserDeBruijn(size)
        q = queue.LifoQueue() #For constructing the tree top to bottom (without weights)
        new_root = cls.node([None] * 4, [1] * 4)  # Will be root node of resulting tree.
        q.put((new_root, height))
        termnode = qmatrix.node(None, None)
        c1 = []
        q_prop = queue.Queue() #Used for traversing back up, propagating factors once weights have been set. Separate queue to not affect building of remaining tree.
        global_weight = 1
        current_leg = 0  # Only used in set_weight(). Keeps track of which bottom leg is being calculated.
        while q.qsize() != 0:
            (curr_node, height) = q.get()
            if height == 1: #Arrived at a leaf node
                legs=z_order_better(current_leg, size, deBruijn) #Gives corresponding matrix indices of the legs of current node
                curr_node.weights = [set_weight(legs[0], size), set_weight(legs[1], size), set_weight(legs[2], size),set_weight(legs[3], size)]
                parents = get_parent_order(legs[0], first) #Gets a list of the parents and the ways to traverse them to get to this leg. Used for propagation of factors
                if parents.qsize() == 1: #Only if matrix is 2*2, no need to propagate
                    curr_node.conns = [termnode] * 4
                    break
                parents.get() #Getting rid of leaf node. Next one is its actual parent.
                (parent,child_index) = parents.get() #This node's parent and index of connection
                current_leg += 4
                curr_node.conns = [termnode] * 4 #Attaching termnode to leaf indicates it has been handled and factors have been propagated.
                if all(weight == 0 for weight in curr_node.weights):
                    curr_node = None
                    propagated_factor = 0
                else:
                    propagated_factor = next((x for x in curr_node.weights if x), None)
                    curr_node.weights = [weight / propagated_factor for weight in curr_node.weights]
                    copy = next((c1_elem for c1_elem in c1 if curr_node == c1_elem), None)
                    if copy is not None:
                        curr_node = copy
                    else:
                        c1.append(curr_node)
                    q_prop.put((parent,child_index, curr_node, propagated_factor)) #Here we start propagating factors up to parents.
                while q_prop.qsize() != 0: #This part deals with parents of parent above leaf node.
                    (curr_node,child_index, child, propagated_factor) = q_prop.get()
                    curr_node.weights[child_index] *= propagated_factor
                    if propagated_factor == 0:
                        curr_node.conns[child_index] = termnode
                    # Now check if factor should be propagated upwards or not.
                    else:
                        # If prop factor is not 0, and all connections of all curr_node's children are None,
                        # a factor has not yet been propagated and this one can be. Once the factor has been propagated,
                        # this node will have a child that is not None and so no further factors will be propagated.
                        if all(curr_node.conns[n] is termnode for n in range(0, child_index)): #If this is the first nonzero propagated factor then propagate
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
                for i in [3, 2, 1, 0]: #The actual building of tree using placeholder nodes.
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
