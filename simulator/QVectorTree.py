from queue import Queue, LifoQueue
from VNode import VNode


class QVectorTree:
    class VNode:
        def __init__(self, child0, child1, w1, w2, depth):
            self.child0 = child0
            self.child1 = child1
            self.w1 = w1
            self.w2 = w2
            self.depth = depth

    def __init__(self, root_node, weight):
        self.root_node = root_node
        self.weight = weight

    # are the methods supposed to be static or called with an instance variable?
    def to_tree(vector_arr):
        # initializing q
        q = Queue(0)

        for weight0, weight1 in pairwise(iter(vector_arr)):  # lump the array in pairs
            node = VNode(None, None, weight0, weight1, 1)  # Create a leaf node from every pair.
            q.put(node)

        while q.qsize() > 1:
            child0 = q.get()
            child1 = q.get()
            depth = max(0 if child0 is None else child0.depth,
                        0 if child1 is None else child1.depth) + 1
            new_node = VNode(child0, child1, 1, 1, depth)
            q.put(new_node)

        node_tree = q.get()

        return QVectorTree(node_tree, 1)

    # returns an array of the values in the leaf nodes.
    # Usage of queue class because its operations put()and get() have-
    # better complexity than regular python lists (O(1) vs O(N))
    def to_vector(tree):
        # Initializing a stack of for all nodes
        s1 = LifoQueue(0)
        # leaf nodes
        s2 = LifoQueue(0)

        # attach rootnode to stack
        s1.put(tree.root_node)

        while s1.qsize() != 0:
            curr = s1.get()

            # If current node has a left child
            # push it onto the first stack
            if curr.child0:
                s1.put(curr.child0)

            # If current node has a right child
            # push it onto the first stack
            if curr.child1:
                s1.put(curr.child1)

            # If current node is a leaf node (Both children are None)
            # push left and right leg-value onto stack
            elif curr.child0 is None and curr.child1 is None:
                s2.put(curr.w1)
                s2.put(curr.w2)

        return s2


def pairwise(iterable):
    # "s -> (s0, s1), (s2, s3), (s4, s5), ..."
    a = iter(iterable)
    return zip(a, a)


# ---------can be removed - for testing purposes-------------
if __name__ == '__main__':
    tree = QVectorTree.to_tree([1, 1, 0, 0])
    vector = QVectorTree.to_vector(tree)
    while vector.not_empty:
        print(vector.get())
