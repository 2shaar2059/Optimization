import numpy as np
import unittest
from ModifiedGramSchmidt import ModifiedGramSchmidt


class Test_ModifiedGramSchmidt(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        print(str("-" * 100))

    def runTest(self, A):
        Q, R = ModifiedGramSchmidt(A.astype(np.float64))

        self.assertTrue(np.all(np.isclose(np.eye(A.shape[0]), Q @ Q.T, atol=1e-10)))
        self.assertTrue(np.all(np.isclose(np.eye(A.shape[0]), Q.T @ Q, atol=1e-10)))
        self.assertTrue(np.all(np.isclose(A, Q @ R, atol=1e-10)))

    def test_Basic(self):
        A = np.array(
            [
                [1, 2, 3],
                [1, 3, 3],
                [2, 3, 4],
            ]
        )
        self.runTest(A)

    def test_poorlyConditioned(self):
        # testcase taken from: https://fgiesen.wordpress.com/2013/06/02/modified-gram-schmidt-orthogonalization/
        epsilon = 1e-5
        A = np.array(
            [
                [1, 1, 1],
                [epsilon, epsilon, 0],
                [epsilon, 0, epsilon],
            ]
        )
        self.runTest(A)


if __name__ == "__main__":
    unittest.main()
