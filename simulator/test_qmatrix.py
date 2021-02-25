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

        newnode, newdepth = qmatrix.node.merge([qm.root, None, None, None], [1,0,0,0])
        qmat = qmatrix(newnode, depth=newdepth)
        result = np.zeros((8,8))
        result[:matrix.shape[0],:matrix.shape[1]] = matrix
        self.assertTrue(np.array_equal(qmat.to_matrix(), result))

if __name__ == '__main__':
    unittest.main()