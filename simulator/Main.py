from qvector import *
from qmatrix import *
from queue import *
import numpy as np
matrix_tree=qmatrix.to_tree(np.matrix(
        [[1,6,4,2],[7,3,357,57],[1,264,6,7],[67,3,2,1]]))
vector_tree = qvector.to_tree((2,7,5,43))
print(qvector.mult(matrix_tree, vector_tree).to_vector())
print(np.matmul(matrix_tree.to_matrix(), vector_tree.to_vector()))
