import gc
import math
from queue import Queue, LifoQueue
import numpy as np


class qvector:
    class node:
        def __init__(self, conns, weights):
            self.conns = conns
            self.weights = weights

        def __hash__(self) -> int:
            return hash((self.conns, self.weights))

        def __eq__(self, o: object) -> bool:
            """Assumes only one copy of earlier nodes exist"""
            return False if not o else self.conns == o.conns and self.weights == o.weights

    def __init__(self, root_node, weight, height):
        self.root_node = root_node
        self.weight = weight
        self.height = height

    @staticmethod
    def to_tree(vector_arr):

        n = len(vector_arr)
        if (n & (n-1) != 0) or n < 2:
            raise ValueError("Array length is not a power of two, length is {}".format(n))
        height = int(math.log2(n))
        # initializing q
        q = Queue(0)

        c1 = [] #List of unique nodes
        for weight0, weight1 in pairwise(iter(vector_arr)):  # lump the array in pairs
            if weight0 == 0 and weight1 == 0:
                node = None
                nonzero = 0
            else:
                nonzero = weight0 if weight0 != 0 else weight1 #Factor for propagation
                normelems = [weight0/nonzero, weight1/nonzero] #Adjust elements
                node = qvector.node([None]*2, normelems)  # Create a leaf node from every pair.
                copy = next((c1_elem for c1_elem in c1 if node == c1_elem), None) #This copies an existing node if it is equal to the one we just created?
                if copy is not None:
                    node = copy
                else:
                    c1.append(node)
            q.put((node, nonzero))

        while q.qsize() > 1:
            node0 = q.get()
            node1 = q.get()
            nodes = (node0[0], node1[0])
            weights = (node0[1], node1[1]) #The weight that has been propagated upwards
            if all(node is None for node in nodes):
                qbc = [None, 0] #qbc will be the next node of one height up?
            else:
                nonzero = next((x for x in weights if x), None)
                normelems = [weight / nonzero for weight in weights]
                qnodeinner = qvector.node(nodes, normelems) #New node
                # TODO: change to something better than O(n) (hash map eq.)
                copyinner = next((c1_elem for c1_elem in c1 if qnodeinner == c1_elem), None) #Check if new node is equivalent to existing one
                if copyinner is not None:
                    qnodeinner = copyinner
                else:
                    c1.append(qnodeinner)
                qbc = [qnodeinner, nonzero]
            q.put(qbc)
        (root, weight) = q.get()

        return qvector(root, weight, height)

    @classmethod
    def mult(cls,matrix_tree,vector_tree):
        def set_weight(current_leg):
            weight = 0
            for i in range(size):
                weight += matrix_tree.get_element_no_touple(current_leg*size+i) * vector_tree.get_element(i)
            return weight

        if (matrix_tree.height != vector_tree.height):
            raise ValueError("Dimensions do not match, mult between ", matrix_tree.to_matrix(), vector_tree.to_vector())

        size = 2 ** matrix_tree.height
        vec_arr_result = []
        for current_leg in range(size):
            vec_arr_result.append(set_weight(current_leg))

        return cls.to_tree(vec_arr_result)

    # returns an array of the values in the leaf nodes.
    # Usage of queue class because its operations put()and get() have-
    # better complexity than regular python lists (O(1) vs O(N))
    def to_vector(self):
        # Initializing a stack of for all nodes
        s1 = LifoQueue(0)
        # leaf nodes
        s2 = []

        # attach rootnode to stack
        s1.put((self.root_node, self.weight, self.height))

        while s1.qsize() != 0:
            curr, weight, height = s1.get()

            if not curr and height > 0:
                s2 += ([0]*(1 << height))
            elif curr:

                # If current node has a right child
                # push it onto the first stack
                s1.put((curr.conns[1], curr.weights[1]*weight, height-1))

                # If current node has a left child
                # push it onto the first stack
                s1.put((curr.conns[0], curr.weights[0]*weight, height-1))

                # If current node is a leaf node (Both conns are None)
                # push left and right leg-value onto stack
                if curr.conns[0] is None and curr.conns[1] is None and height == 1:
                    s2.append(round(curr.weights[0]*weight,7))
                    s2.append(round(curr.weights[1]*weight,7))

        return s2

    def get_element(self, index):
        size = 1 << self.height-1
        if (index >= size << 1 or index < 0):
            raise ValueError("Index out of bounds, index was {} when allowed values are 0 - {}".format(index, size - 1))
        value = self.weight
        target = self.root_node
        while size > 0:
            goto = 0
            if index & size:
                goto += 1
            if target.weights[goto] == 0:
                return 0
            value *= target.weights[goto]
            target = target.conns[goto]
            size = size >> 1
        return value

    def measure(self):
        vector=self.to_vector()
        res = [round(abs(x)**2,7) for x in vector]
        return res

    def measureSingle(self, qubit): #Not tested much, but also not necessary?
        import random
        def collapse(outcome, normalization):
            if outcome:
                not_outcome = 0
            else:
                not_outcome = 1

            q = LifoQueue()
            q.put((self.root_node , 0)) # Second part of tuple is the current qubit ( depth ). Starts q_0

            while q.qsize() != 0:
                (node,depth) = q.get()
                if(depth < qubit):
                    for i in [0,1]:
                        q.add((node.conns(i), depth + 1))
                else:
                    node.conns[not_outcome] = None #Why are these touples again? Isn't that kind of annoying
                    node.weights[not_outcome] = 0
                    node.weights[outcome] /= normalization

        if qubit > self.height:
            raise ValueError("The qubit ", qubit, "does not exist in the vector of height ", self.height)

        size = 1 << self.height
        sub_vectors = 1 << (qubit + 1)
        sub_vector_size = size//sub_vectors
        probability_zero = 0
        for i in range(sub_vectors//2): #This can be done in parallell?
            for j in range(sub_vector_size):
                probability_zero += self.get_element(i*sub_vector_size*2+j)**2 #Is the method in the paper more efficient?
                #probability_one += self.get_element(i*sub_vector_size*2+j+sub_vector_size)**2 #Not needed because = 1 - probability_zero ?

        random = random.random() #Float between 0 and 1
        if random <= probability_zero:
            collapse(0, math.sqrt(probability_zero)) # Should we check if the vector is still normalized for testing purposes?
            return 0 #No need to return?
        else:
            collapse(1, math.sqrt(1-probability_zero))
            return 1 #No need to return?

def pairwise(iterable):
    # "s -> (s0, s1), (s2, s3), (s4, s5), ..."
    a = iter(iterable)
    return zip(a, a)


# ---------can be removed - for testing purposes-------------
if __name__ == '__main__':
    tree = qvector.to_tree([1, 2, 4, 8, 0, 0, 0, 0])
    vector = tree.to_vector()
    print(vector)
