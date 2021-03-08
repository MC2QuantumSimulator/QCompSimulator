import unittest
import numpy as np
from qmatrix import qmatrix

class TestQmatrix(unittest.TestCase):

    def test_to_matrix(self):
        qn1 = qmatrix.node([None]*4, [1,2, 3,4])
        self.assertTrue(np.array_equal(qmatrix(qn1).to_matrix(), np.array([[1,2], [3,4]])))

    def test_to_tree(self):
        matrix = np.arange(16).reshape((4, 4))
        qm = qmatrix.to_tree(matrix)
        self.assertTrue(np.array_equal(qm.to_matrix(), matrix))

        newnode, newheight = qmatrix.node.merge([qm.root, None, None, None], [2,0,0,0])
        qmat = qmatrix(newnode, height=newheight)
        result = np.zeros((8,8))
        result[:matrix.shape[0],:matrix.shape[1]] = matrix
        self.assertTrue(np.array_equal(qmat.to_matrix(), result))

    def test_kron(self):
        # Assume to_matrix and to_tree work correctly:
        mat1 = np.array([[1,0],[0,1]])
        mat2 = np.array([[1,1],[1,-1]])
        qmat1 = qmatrix.to_tree(mat1)
        qmat2 = qmatrix.to_tree(mat2)
        matkron = np.kron(mat1, mat2)
        qmatkron = qmatrix.kron(qmat1, qmat2)
        self.assertTrue(np.array_equal(qmatkron.to_matrix(), matkron))

    def test_mult(self):
        first = qmatrix.to_tree(np.matrix(
            [[1, 6, 4, 2], [7, 3, 357, 57], [1, 264, 6, 7], [67, 3, 2, 1]]))
        second = qmatrix.to_tree(np.matrix(
            [[7, 5, 4, 23], [7, 34, 45, 6], [23, 7, 8, 6], [76, 89, 32, 2]]))
        self.assertTrue(np.array_equal(qmatrix.mult(first, second).to_matrix(),
                                       np.matmul(first.to_matrix(), second.to_matrix())))

    def test_mult1(self):
        first = qmatrix.to_tree(np.matrix([[1, 2], [3, 4]]))
        second = qmatrix.to_tree(np.matrix([[1, 2], [3, 4]]))
        self.assertTrue(np.array_equal(qmatrix.mult(first, second).to_matrix(),
                                       np.matmul(first.to_matrix(), second.to_matrix())))

    def test_mult2(self):
        first = qmatrix.to_tree(np.matrix(
        [[1, 2, 7, 2, 4, 5, 3, 2], [4, 7, 8, 3, 5, 4, 1, 9], [1, 7, 45, 6, 8, 9, 4, 7], [1, 6, 2, 3, 4, 4, 3, 2],
         [1, 2, 7, 1, 4, 5, 7, 2], [4, 2, 8, 4, 5, 4, 1, 9], [1, 7, 89, 6, 5, 9, 4, 7], [1, 6, 6, 1, 4, 7, 3, 2]]))
        second = qmatrix.to_tree(np.matrix(
        [[54,7,8,2,23,5,7,8], [6,43,3,9,7,6,4,2], [76,8,3,2,43,8,9,3], [6,9,3,3,1,7,54,2354],
         [67,8,3,45,8,8,5,2], [7,4,4,2,6,87,9,54], [7,32,5,8,9,5,432,12], [6,8,56,3,2,6,8,9]]))

if __name__ == '__main__':
    unittest.main()