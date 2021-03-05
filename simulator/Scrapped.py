@classmethod
def mult(cls, matrix_tree, vector_tree):  # Matrix - vector multiplication
    # First, calculate the values of the bottom legs.
    # This is stupid version which is basically regular matrix multiplication. But is there actually a way to make it better?
    if matrix_tree.depth != vector_tree.depth:
        raise ValueError("Only matrixes and trees with the same depth can be multiplied.")
    size = 1 << (matrix_tree.depth + 1)

    bottom_legs = [None] * (size)
    vector = vector_tree.to_vector()  # TODO: Stay in tree form instead
    sum = 0
    for i in range(size):
        for j in range(size):
            sum += matrix_tree.get_element([i, j]) * vector[j]
        bottom_legs[i] = sum
        sum = 0
    return cls.to_tree(bottom_legs)


@classmethod
def mult_better(cls, matrix_tree, vector_tree):
    if matrix_tree.depth != vector_tree.depth:
        raise ValueError("Only matrixes and trees with the same depth can be multiplied.")
    size = 1 << (matrix_tree.depth + 1)

    q = Queue(0)  # First in first out queue.
    curr_matrix_node = matrix_tree.root
    curr_vector_node = vector_tree.root
    curr_new_node = qvector.node([None] * 2, (0, 0))  # Will be the root of the new vector, post multiplication.
    q.put(curr_matrix_node, curr_vector_node, curr_new_node,
          0)  # Putting the matrix and vector qubit that will be multiplied, followed by its depth.

    for elem in range(size):
        q.put(qvector.node([None] * 2, (0, 0)))

    while q.size() > 0:
        (curr_matrix_node, curr_vector_node, curr_new_node) = q.get()
        if all(conn is None for conn in curr_matrix_node.conns):
            curr_new_node.weights = ((curr_matrix_node.weights[0] * curr_vector_node.weights[0] +
                                      curr_matrix_node.weights[1] * curr_vector_node.weights[1]),
                                     (curr_matrix_node.weights[3] * curr_vector_node.weights[0] +
                                      curr_matrix_node.weights[3] * curr_vector_node.weights[1]))
            curr_new_node.children = (None, None)
        else:
            new_node1 = qvector.node([None] * 2, (1, 1))
            new_node2 = qvector.node([None] * 2, (1, 1))
            q.put(new_node1)
            q.put(new_node2)


@classmethod
def add_trees(cls, vector_tree0, vector_tree1):
    q = Queue(0)
    curr_node0 = vector_tree0.root_node
    curr_node1 = vector_tree1.root_node
    # q.put(curr_node0.children[0], curr_node1.children[0], 0)
    # q.put(curr_node0.children[1], curr_node1.children[1], 1)
    q.put(curr_node0.children[0])
    q.put(curr_node1.children[0])
    q.put(0)
    q.put(curr_node0.children[1])
    q.put(curr_node1.children[1])
    q.put(1)
    new_root = qvector.node([None] * 2, (
    curr_node0.weights[0] + curr_node1.weights[0], curr_node0.weights[1] + curr_node1.weights[1]))
    prev_node = new_root

    while q.qsize() > 0:  # All nodes of first tree are put in a queue
        # (curr_node0, curr_node1, side)=q.get()

        curr_node0 = q.get()
        curr_node1 = q.get()
        side = q.get()
        new_node = qvector.node([None] * 2, (
        curr_node0.weights[0] + curr_node1.weights[0], curr_node0.weights[1] + curr_node1.weights[1]))
        print(curr_node0.weights[0], curr_node1.weights[0], curr_node0.weights[1], curr_node1.weights[1])
        print((curr_node0.weights[0] + curr_node1.weights[0], curr_node0.weights[1] + curr_node1.weights[1]))
        prev_node.children[side] = (new_node)
        print("new node children: %s, %s" % (new_node.children[0], new_node.children[1]))

        if (side == 1):
            prev_node = new_node

        if all(children is None for children in curr_node0.children):
            print("Passing")
            pass
        else:
            print("adding")
            # q.put(curr_node0.children[0], curr_node1.children[0], 0)
            # q.put(curr_node0.children[1], curr_node1.children[1], 1)
            q.put(curr_node0.children[0])
            q.put(curr_node1.children[0])
            q.put(0)
            q.put(curr_node0.children[1])
            q.put(curr_node1.children[1])
            q.put(1)

    print(new_root.children[1])
    return qvector(new_root, (vector_tree0.weight + vector_tree1.weight), vector_tree0.depth)


    curr_node.weights=((matrix_tree.get_element_no_touple(qvector.sub_matrix_indexing(matrix_index,matrix_tree.height))*vector_tree.get_element(vector_index)+\
                                        matrix_tree.get_element_no_touple(qvector.sub_matrix_indexing(matrix_index+1,matrix_tree.height))*vector_tree.get_element(vector_index+1),\
                                        matrix_tree.get_element_no_touple(qvector.sub_matrix_indexing(matrix_index+2,matrix_tree.height))*vector_tree.get_element(vector_index)+\
                                        matrix_tree.get_element_no_touple(qvector.sub_matrix_indexing(matrix_index+3,matrix_tree.height))*vector_tree.get_element(vector_index+1)))