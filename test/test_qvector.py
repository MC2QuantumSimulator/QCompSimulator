import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import unittest
from qvector import qvector
import numpy as np
from qmatrix import qmatrix

class TestQvector(unittest.TestCase):

    def test_to_vector(self):
        # TODO
        pass

    def test_to_tree(self):
        # TODO
        pass

    def test_mult(self):
        matrix_tree = qmatrix.to_tree(np.array(
        [[1, 2, 7, 2, 4, 5, 3, 2], [4, 7, 8, 3, 5, 4, 1, 9], [1, 7, 45, 6, 8, 9, 4, 7], [1, 6, 2, 3, 4, 4, 3, 2],
         [1, 2, 7, 1, 4, 5, 7, 2], [4, 2, 8, 4, 5, 4, 1, 9], [1, 7, 89, 6, 5, 9, 4, 7], [1, 6, 6, 1, 4, 7, 3, 2]]))
        vector_tree = qvector.to_tree((1, 6, 8, 3, 5, 4, 6, 1))
        self.assertTrue(np.array_equal(qvector.mult(matrix_tree,vector_tree).to_vector(),
                                       np.matmul(matrix_tree.to_matrix(),vector_tree.to_vector())))
    def test_mult1(self):
        matrix_tree=qmatrix.to_tree(np.array(
        [[1,6,4,2],[7,3,357,57],[1,264,6,7],[67,3,2,1]]))
        vector_tree = qvector.to_tree((2,7,5,43))
        self.assertTrue(np.array_equal(qvector.mult(matrix_tree, vector_tree).to_vector(),
                                       np.matmul(matrix_tree.to_matrix(), vector_tree.to_vector())))

if __name__ == '__main__':
    unittest.main()