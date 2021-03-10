from queue import Queue, LifoQueue
import numpy as np


class qvector:
    class node:
        def __init__(self, conns, weights):
            self.conns = conns
            self.weights = weights

        def __eq__(self, o: object) -> bool:
            """Assumes only one copy of earlier nodes exist"""
            return self.conns == o.conns and self.weights == o.weights

    def __init__(self, root_node, weight, height):
        self.root_node = root_node
        self.weight = weight
        self.height = height

    @staticmethod
    def to_tree(vector_arr):
        # initializing q
        q = Queue(0)

        c1 = []
        for weight0, weight1 in pairwise(iter(vector_arr)):  # lump the array in pairs
            if weight0 == 0 and weight1 == 0:
                node = None
                nonzero = 0
            else:
                nonzero = weight0 if weight0 != 0 else weight1
                normelems = [weight0/nonzero, weight1/nonzero]
                node = qvector.node([None]*2, normelems)  # Create a leaf node from every pair.
                copy = next((c1_elem for c1_elem in c1 if node == c1_elem), None)
                if copy is not None:
                    node = copy
                else:
                    c1.append(node)
            q.put((node, nonzero, 1))

        while q.qsize() > 1:
            node0 = q.get()
            node1 = q.get()
            nodes = (node0[0], node1[0])
            weights = (node0[1], node1[1])
            heights = (node0[2], node1[2])
            if all(node is None for node in nodes):
                qbc = [None, 0, 1]
            else:
                nonzero = next((x for x in weights if x), None)
                normelems = [weight / nonzero for weight in weights]
                qnodeinner = qvector.node(nodes, normelems)
                height = max([1 if height is None else height for height in heights]) + 1
                # TODO: change to something better than O(n) (hash map eq.)
                copyinner = next((c1_elem for c1_elem in c1 if qnodeinner == c1_elem), None)
                if copyinner is not None:
                    qnodeinner = copyinner
                else:
                    c1.append(qnodeinner)
                qbc = [qnodeinner, nonzero, height]
            q.put(qbc)
        (root, weight, height) = q.get()

        return qvector(root, weight, height)

    @classmethod
    def mult(cls,matrix_tree,vector_tree):
        #Plan: Create node from the top with no childsm and put it in a queue. Take node from queue and check if it's childless. If it is, create childs of lower height and put in queue. Get from queue ( last in first out ) same node and check if childless,
        #if yes, create child. When at depth 1, set weights. Doing it this way should finish one side of the tree first. When weights have been set, can start propagating factors.
        current_leg=0
        def set_weight(current_leg):
            weight = 0
            for i in range(size):
                weight += matrix_tree.get_element_no_touple(current_leg*size+i) * vector_tree.get_element(i)
            return weight

        if (matrix_tree.height != vector_tree.height):
            raise ValueError("Dimensions do not match, mult between ", matrix_tree.to_matrix(), vector_tree.to_vector())

        q=LifoQueue()
        height=matrix_tree.height
        size=2**height
        new_root=cls.node([None]*2,(1,1)) #Will be root node of resulting tree.
        q.put((new_root , height))

        while q.qsize() != 0:
            (curr_node,height)=q.get()
            if height==1:
                curr_node.weights=(set_weight(current_leg),set_weight(current_leg+1))
                current_leg+=2
                #A "sub tree" should be finished at this point. Possibly insert some cleanup here?
            else:
                for i in [1,0]:
                    if curr_node.conns[i] is None:
                        new_node=cls.node([None]*2,(1,1))
                        curr_node.conns[i]=new_node
                        q.put((new_node,height-1))

        return qvector(new_root,1,matrix_tree.height)

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
                    s2.append(curr.weights[0]*weight)
                    s2.append(curr.weights[1]*weight)

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

def pairwise(iterable):
    # "s -> (s0, s1), (s2, s3), (s4, s5), ..."
    a = iter(iterable)
    return zip(a, a)


# ---------can be removed - for testing purposes-------------
if __name__ == '__main__':
    tree = qvector.to_tree([1, 2, 4, 8, 0, 0, 0, 0])
    vector = tree.to_vector()
    print(vector)
