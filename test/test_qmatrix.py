import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import unittest
import numpy as np
from simulator.qmatrix import qmatrix

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
        self.assertTrue(np.allclose(qmatkron.to_matrix(), matkron))

    def test_mult(self):
        first = qmatrix.to_tree(np.array(
            [[1, 6, 4, 2], [7, 3, 357, 57], [1, 264, 6, 7], [67, 3, 2, 1]]))
        second = qmatrix.to_tree(np.array(
            [[7, 5, 4, 23], [7, 34, 45, 6], [23, 7, 8, 6], [76, 89, 32, 2]]))
        self.assertTrue(np.allclose(qmatrix.mult(first, second).to_matrix(),
                                       np.matmul(first.to_matrix(), second.to_matrix())))

    def test_mult1(self):
        first = qmatrix.to_tree(np.array([[1, 2], [3, 4]]))
        second = qmatrix.to_tree(np.array([[1, 2], [3, 4]]))
        self.assertTrue(np.allclose(qmatrix.mult(first, second).to_matrix(),
                                       np.matmul(first.to_matrix(), second.to_matrix())))

    def test_mult2(self):
        first = qmatrix.to_tree(np.array(
        [[1, 2, 7, 2, 4, 5, 3, 2], [4, 7, 8, 3, 5, 4, 1, 9], [1, 7, 45, 6, 8, 9, 4, 7], [1, 6, 2, 3, 4, 4, 3, 2],
         [1, 2, 7, 1, 4, 5, 7, 2], [4, 2, 8, 4, 5, 4, 1, 9], [1, 7, 89, 6, 5, 9, 4, 7], [1, 6, 6, 1, 4, 7, 3, 2]]))
        second = qmatrix.to_tree(np.array(
        [[54,7,8,2,23,5,7,8], [6,43,3,9,7,6,4,2], [76,8,3,2,43,8,9,3], [6,9,3,3,1,7,54,2354],
         [67,8,3,45,8,8,5,2], [7,4,4,2,6,87,9,54], [7,32,5,8,9,5,432,12], [6,8,56,3,2,6,8,9]]))
        self.assertTrue(np.allclose(qmatrix.mult(first, second).to_matrix(),np.matmul(first.to_matrix(), second.to_matrix())))

    def test_mult3(self):
        twoid = qmatrix.to_tree(np.array([[1, 0], [0, 1]]))
        had = qmatrix.to_tree(np.array(
            [[1, 1], [1, -1]]))
        fourid = qmatrix.to_tree(np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]))
        first = qmatrix.to_tree(np.array(
            [[1, 2, 7, 2, 4, 5, 3, 2], [4, 7, 8, 3, 5, 4, 1, 9], [1, 7, 45, 6, 8, 9, 4, 7], [1, 6, 2, 3, 4, 4, 3, 2],
             [1, 2, 7, 1, 4, 5, 7, 2], [4, 2, 8, 4, 5, 4, 1, 9], [1, 7, 89, 6, 5, 9, 4, 7], [1, 6, 6, 1, 4, 7, 3, 2]]))
        second = qmatrix.to_tree(np.array(
            [[54, 7, 8, 2, 23, 5, 7, 8], [6, -43, 3, 9, -7, 6, 4, 2], [76, -8, 3, 2, 43, 8, -9, 3],
             [6, -9, 3, 3, 1, 7, 54, 2354],
             [67, -8, 3, 45, 8, 8, 5, 2], [7, 4, 4, 2, 6, 87, 9, 54], [7, 32, 5, 8, 9, 5, 432, 12],
             [6, 8, 56, 3, 2, 6, 8, 9]]))
        eightid = qmatrix.to_tree(np.array(
            [[1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0],
             [0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 1]]))
        third = qmatrix.to_tree(np.array(
            [[1, 6, 4, -2], [7, 3, 357, 57], [-1, 264, 6, 7], [67, -3, 2, 1]]))
        self.assertTrue(np.allclose(qmatrix.mult(eightid, first).to_matrix(), np.matmul(eightid.to_matrix(), first.to_matrix())))
        self.assertTrue(np.allclose(qmatrix.mult(first, eightid).to_matrix(), np.matmul(first.to_matrix(), eightid.to_matrix())))
        self.assertTrue(np.allclose(qmatrix.mult(second, eightid).to_matrix(), np.matmul(second.to_matrix(), eightid.to_matrix())))
        self.assertTrue(np.allclose(qmatrix.mult(eightid, second).to_matrix(), np.matmul(eightid.to_matrix(), second.to_matrix())))
        self.assertTrue(np.allclose(qmatrix.mult(third, fourid).to_matrix(), np.matmul(third.to_matrix(), fourid.to_matrix())))
        res = qmatrix.kron(had, fourid)
        fourid = qmatrix.to_tree(np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]))
        had = qmatrix.to_tree(np.array(
            [[1, 1], [1, -1]]))
        self.assertTrue(np.allclose(qmatrix.mult(eightid, res).to_matrix(), np.matmul(eightid.to_matrix(), res.to_matrix())))
        res = qmatrix.kron(twoid, had)
        self.assertTrue(np.allclose(qmatrix.mult(fourid, res).to_matrix(), np.matmul(fourid.to_matrix(), res.to_matrix())))
        
if __name__ == '__main__':
    unittest.main()