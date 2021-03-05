from queue import Queue, LifoQueue
import numpy as np


class qvector:
    class node:
        def __init__(self, conns, weights):
            self.conns = conns
            self.weights = weights

    def __init__(self, root_node, weight, height):
        self.root_node = root_node
        self.weight = weight
        self.height = height

    @staticmethod
    def to_tree(vector_arr):
        # initializing q
        q = Queue(0)

        for weight0, weight1 in pairwise(iter(vector_arr)):  # lump the array in pairs
            node = qvector.node([None]*2, (weight0, weight1))  # Create a leaf node from every pair.
            q.put((node, 0))

        while q.qsize() > 1:
            node0 = q.get()
            node1 = q.get()
            conns = (node0[0], node1[0])
            heights = (node0[1], node1[1])
            height = max(0 if height is None else height for height in heights) + 1
            new_node = qvector.node(conns, (1, 1))
            q.put((new_node, height))

        node_tree, height = q.get()

        return qvector(node_tree, 1, height)

    @classmethod
    def mult(cls,matrix_tree,vector_tree):
        #Plan: Create node from the top with no childsm and put it in a queue. Take node from queue and check if it's childless. If it is, create a child of lower height and put in queue. Get from queue ( last in first out ) same node and check if childless,
        #if yes, create child. When at depth 1, set weights. Doing it this way should finish one side of the tree first. When weights have been set, can start propagating factors.
        q=LifoQueue()
        height=matrix_tree.height
        new_root=cls.node([None]*2,(1,1))
        q.put((new_root , height))
        matrix_index=0
        while q.qsize() != 0:
            (curr_node,height)=q.get()
            if height==1:
                weight_left=0
                weight_right=0
                vector_index=0
                for i in range(2**matrix_tree.height):
                    weight_left+=matrix_tree.get_element_no_touple(matrix_index)*vector_tree.get_element(vector_index)
                    matrix_index+=1
                    vector_index+=1
                vector_index=0
                for i in range(2**matrix_tree.height):
                    weight_right+=matrix_tree.get_element_no_touple(matrix_index)*vector_tree.get_element(vector_index)
                    matrix_index+=1
                    vector_index+=1
                curr_node.weights=(weight_left,weight_right)
            else:
                i=1
                while i != -1:
                    if curr_node.conns[i] is None:
                        new_node=cls.node([None]*2,(1,1))
                        curr_node.conns[i]=new_node
                        q.put((new_node,height-1))
                    i-=1
        return qvector(new_root,1,matrix_tree.height)

    def get_element(self,element):
        vector = self.to_vector()
        return vector[element]

    @classmethod
    def add(cls,vector_tree1,vector_tree2): #REDUNDANT?
        bottom_nodes=np.add(vector_tree1.to_vector(),vector_tree2.to_vector())
        return cls.to_tree(bottom_nodes)

    # returns an array of the values in the leaf nodes.
    # Usage of queue class because its operations put()and get() have-
    # better complexity than regular python lists (O(1) vs O(N))
    def to_vector(self):
        # Initializing a stack of for all nodes
        s1 = LifoQueue(0)
        # leaf nodes
        s2 = []

        # attach rootnode to stack
        s1.put(self.root_node)

        while s1.qsize() != 0:
            curr = s1.get()

            # If current node has a right child
            # push it onto the first stack
            if curr.conns[1]:
                s1.put(curr.conns[1])

            # If current node has a left child
            # push it onto the first stack
            if curr.conns[0]:
                s1.put(curr.conns[0])

            # If current node is a leaf node (Both conns are None)
            # push left and right leg-value onto stack
            elif curr.conns[0] is None and curr.conns[1] is None:
                s2.append(curr.weights[0])
                s2.append(curr.weights[1])

        return s2

    @staticmethod
    def sub_matrix_indexing(input_index, qubits): #REDUNDANT?
        size = 1 << qubits
        q = Queue()
        list_matrix = []
        for i in range(size ** 2):
            list_matrix.append(i)
        elements = []
        q.put((list_matrix, size))

        while q.qsize() != 0:
            (curr_matrix, size) = q.get()
            size_half = size // 2
            if size == 2:
                for i in range(4):
                    elements.append(curr_matrix[i])
            else:
                sub = 0
                for elem in range(4):
                    sub_matrix = []
                    for i in range(size_half):
                        for j in range(size_half):
                            index = j + size * i + sub
                            sub_matrix.append(curr_matrix[index])
                    if elem == 1:
                        sub = (size ** 2) // 2
                    else:
                        sub += size_half
                    q.put((sub_matrix, size // 2))
        return elements.index(input_index)

    def get_element(self, index):
        size = 1 << self.height
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

def pairwise(iterable):
    # "s -> (s0, s1), (s2, s3), (s4, s5), ..."
    a = iter(iterable)
    return zip(a, a)


# ---------can be removed - for testing purposes-------------
if __name__ == '__main__':
    tree = qvector.to_tree([1, 2, 3, 4, 5, 6, 7, 8])
    vector = tree.to_vector()
    print(vector)
