import gc
import math
from queue import Queue, LifoQueue
import numpy as np
from functools import lru_cache
import sys


class qvector:

    # Recursion limit can be changed
    sys.setrecursionlimit(1500)
    cache_size = 1024

    class node:
        def __init__(self, conns, weights):
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

    def __init__(self, root, weight, height):
        self.root = root
        self.weight = weight
        self.height = height

    @classmethod
    @lru_cache(maxsize=cache_size)
    def cache_node(cls, conns, weights):
        return cls.node(conns, weights)

    @classmethod
    @lru_cache(maxsize=cache_size)
    def add_nodes(cls, first: node, second: node, height: int, weights_parent: tuple) -> node:
        if not first and not second:
            return (None, 0)
        if not first:
            firstconns = (None, None)
            firstweights = (0, 0)
        else:
            firstconns = first.conns
            firstweights = first.weights
        if not second:
            secondconns = (None, None)
            secondweights = (0, 0)
        else:
            secondconns = second.conns
            secondweights = second.weights
        if height > 1:
            conns_n_weights = tuple([cls.add_nodes(x, y, height-1, (z*weights_parent[0], w*weights_parent[1])) for x,y,z,w in zip(firstconns, secondconns, firstweights, secondweights)])
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
            node = cls.cache_node(conns, normelems)
            return (node, nonzero)

        nonzero = next((x for x in weights_from_children if x), 1)
        normelems = tuple([weight / nonzero for weight in weights_from_children])
        
        node = cls.cache_node(conns, normelems)
        return (node, nonzero)

    @classmethod
    def add_vectors(cls, first, second):
        if not first.root:
            return second
        if not second.root:
            return first
        # TODO: nuke first and second
        # Used for debugging the add_nodes func, not needed for sim
        new_node, norm = cls.add_nodes(first.root, second.root, first.height, (first.weight,second.weight))
        return cls(new_node, norm, first.height)

    @classmethod
    @lru_cache(maxsize=cache_size)
    def mult_nodes(cls, first: node, second: node, height: int, weight_from_parent: float) -> node:
        if not first or not second:
            return (None, 0)
        newweightsleft = tuple([first.weights[x]*second.weights[y] for x,y in zip((0,2),(0,0))])
        newweightsright = tuple([first.weights[x]*second.weights[y] for x,y in zip((1,3),(1,1))])
        if height <= 1:
            retweights = tuple([x+y for x,y in zip(newweightsleft, newweightsright)])
            nonzero = next((x for x in retweights if x), 1)
            normelems = tuple([weight / nonzero for weight in retweights])
            node = cls.cache_node((None, None), normelems)
            return (node, nonzero*weight_from_parent)
        newconnsleft_n_weights = tuple([cls.mult_nodes(first.conns[x], second.conns[y], height-1, first.weights[x]*second.weights[y]) for x,y in zip((0,2),(0,0))])
        newconnsright_n_weights = tuple([cls.mult_nodes(first.conns[x], second.conns[y], height-1, first.weights[x]*second.weights[y]) for x,y in zip((1,3),(1,1))])
        newconnsleft = tuple([None if not item else item[0] for item in newconnsleft_n_weights])
        newconnsright = tuple([None if not item else item[0] for item in newconnsright_n_weights])
        weightpropleft = tuple([0 if not item else item[1] for item in newconnsleft_n_weights])
        weightpropright = tuple([0 if not item else item[1] for item in newconnsright_n_weights])
        newleft = cls.node(newconnsleft, weightpropleft)
        newright = cls.node(newconnsright, weightpropright)
        result, weight = cls.add_nodes(newleft, newright, height, (1, 1))
        return (result, weight*weight_from_parent)

    @classmethod
    def mult(cls, first, second):
        # TODO: nuke first and second
        new_node, weight = cls.mult_nodes(first.root, second.root, first.height, first.weight*second.weight)
        return cls(new_node, weight, first.height)

    @staticmethod
    def to_tree(vector_arr):

        n = len(vector_arr)
        if (n & (n-1) != 0) or n < 2:
            raise ValueError("Array length is not a power of two, length is {}".format(n))
        height = int(math.log2(n))
        # initializing q
        q = Queue(0)

        c1 = set() #List of unique nodes
        for weight0, weight1 in pairwise(iter(vector_arr)):  # lump the array in pairs
            if weight0 == 0 and weight1 == 0:
                node = None
                nonzero = 0
            else:
                nonzero = weight0 if weight0 != 0 else weight1 #Factor for propagation
                normelems = (weight0/nonzero, weight1/nonzero) #Adjust elements
                node = qvector.node((None, None), normelems)  # Create a leaf node from every pair.
                copy = next((c1_elem for c1_elem in c1 if node == c1_elem), None) #This copies an existing node if it is equal to the one we just created?
                if copy is not None:
                    node = copy
                else:
                    c1.add(node)
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
                normelems = tuple([weight / nonzero for weight in weights])
                qnodeinner = qvector.node(nodes, normelems) #New node
                # TODO: change to something better than O(n) (hash map eq.)
                copyinner = next((c1_elem for c1_elem in c1 if qnodeinner == c1_elem), None) #Check if new node is equivalent to existing one
                if copyinner is not None:
                    qnodeinner = copyinner
                else:
                    c1.add(qnodeinner)
                qbc = [qnodeinner, nonzero]
            q.put(qbc)
        (root, weight) = q.get()

        return qvector(root, weight, height)

    # returns an array of the values in the leaf nodes.
    # Usage of queue class because its operations put()and get() have-
    # better complexity than regular python lists (O(1) vs O(N))
    def to_vector(self):
        # Initializing a stack of for all nodes
        s1 = LifoQueue(0)
        # leaf nodes
        s2 = []

        # attach rootnode to stack
        s1.put((self.root, self.weight, self.height))

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
        target = self.root
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
            q.put((self.root , 0)) # Second part of tuple is the current qubit ( depth ). Starts q_0

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

    def number_of_nodes(self):
        "Returns the number of unique nodes in the tree"
        # LIFO queue to store nodes to be traversed
        s1 = LifoQueue()
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
    def zero_state(cls, n):
        "Returns a Qvector of height n representing the |0> state"
        # Create the first layer
        zero_node = cls.node((None,None), (1,0))
        # Add n-1 more layers, pointing at the new result each time
        for _ in range(n-1):
            zero_node = cls.node((zero_node,None), (1,0))
        return cls(zero_node, 1, n)

def pairwise(iterable):
    # "s -> (s0, s1), (s2, s3), (s4, s5), ..."
    a = iter(iterable)
    return zip(a, a)


# ---------can be removed - for testing purposes-------------
if __name__ == '__main__':
    tree = qvector.to_tree([1, 2, 4, 8, 0, 0, 0, 0])
    vector = tree.to_vector()
    print(vector)
