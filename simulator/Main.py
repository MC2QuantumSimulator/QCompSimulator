from qvector import *
from qmatrix import *
from queue import *
import numpy as np
matrix_tree=qmatrix.to_tree(np.matrix([[1,2,7,2,4,5,3,2],[4,7,8,3,5,4,1,9],[1,7,45,6,8,9,4,7],[1,6,2,3,4,4,3,2],
                                       [1,2,7,2,4,5,3,2],[4,7,8,3,5,4,1,9],[1,7,45,6,8,9,4,7],[1,6,6,3,4,4,3,2]]))
vector_tree=qvector.to_tree((1,6,8,3,5,4,6,1))
print(qvector.mult(matrix_tree,vector_tree).to_vector())
print(np.matmul(matrix_tree.to_matrix(),vector_tree.to_vector()))
