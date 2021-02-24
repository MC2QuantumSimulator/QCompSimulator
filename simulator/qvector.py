from queue import Queue, LifoQueue


class qvector:
    class VNode:
        def __init__(self, children, weights, depth):
            self.children = children
            self.weights = weights
            self.depth = depth

    def __init__(self, root_node, weight):
        self.root_node = root_node
        self.weight = weight

    @staticmethod
    def to_tree(vector_arr):
        # initializing q
        q = Queue(0)

        for weight0, weight1 in pairwise(iter(vector_arr)):  # lump the array in pairs
            node = qvector.VNode((None, None), (weight0, weight1), 1)  # Create a leaf node from every pair.
            q.put(node)

        while q.qsize() > 1:
            child0 = q.get()
            child1 = q.get()
            depth = max(0 if child0 is None else child0.depth,
                        0 if child1 is None else child1.depth) + 1
            new_node = qvector.VNode((child0, child1), (1, 1), depth)
            q.put(new_node)

        node_tree = q.get()

        return qvector(node_tree, 1)

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
            if curr.children[1]:
                s1.put(curr.children[1])

            # If current node has a left child
            # push it onto the first stack
            if curr.children[0]:
                s1.put(curr.children[0])

            # If current node is a leaf node (Both children are None)
            # push left and right leg-value onto stack
            elif curr.children[0] is None and curr.children[1] is None:
                s2.append(curr.weights[0])
                s2.append(curr.weights[1])

        return s2


def pairwise(iterable):
    # "s -> (s0, s1), (s2, s3), (s4, s5), ..."
    a = iter(iterable)
    return zip(a, a)


# ---------can be removed - for testing purposes-------------
if __name__ == '__main__':
    tree = qvector.to_tree([1, 2, 3, 4, 5, 6, 7, 8])
    vector = tree.to_vector()
    print(vector)
