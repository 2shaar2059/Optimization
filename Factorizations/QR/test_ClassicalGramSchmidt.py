import numpy as np
import unittest
from ClassicGramSchmidt import ClassicGramSchmidt


class Test_ClassicGramSchmidt(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        print(str("-" * 100))

    def runTest(self, A, ortho_tol, reconstruction_tolerance):
        """runs the test case

        Args:
            A: matrix to factorize
            ortho_tol: orthogonality tolerance
            reconstruction_tolerance: reconstruction tolerance
        """
        Q, R = ClassicGramSchmidt(A.astype(np.float64))
        self.assertTrue(np.all(np.isclose(np.eye(A.shape[0]), Q @ Q.T, atol=ortho_tol)))
        self.assertTrue(np.all(np.isclose(np.eye(A.shape[0]), Q.T @ Q, atol=ortho_tol)))
        self.assertTrue(np.all(np.isclose(A, Q @ R, atol=reconstruction_tolerance)))

    def test_Basic(self):
        A = np.array(
            [
                [1, 2, 3],
                [1, 3, 3],
                [2, 3, 4],
            ]
        )
        self.runTest(A, 1e-14, 1e-10)

    def test_poorlyConditioned(self):
        # testcase taken from: https://fgiesen.wordpress.com/2013/06/02/modified-gram-schmidt-orthogonalization/
        epsilon = 1e-6
        A = np.array(
            [
                [1, 1, 1],
                [epsilon, epsilon, 0],
                [epsilon, 0, epsilon],
            ]
        )
        # poorly conditioned matrix requires looser tolerance to pass orthogonality checks
        self.runTest(A, 1e-3, 1e-10)


if __name__ == "__main__":
    unittest.main()
