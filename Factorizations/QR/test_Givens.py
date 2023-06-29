import time
import numpy as np
import unittest
from Givens import Givens


class Test_Givens(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        print(str("-" * 100))

    def runTest(self, A):
        Q, R = Givens(A.astype(np.float64))

        self.assertTrue(np.all(np.isclose(np.eye(A.shape[0]), Q @ Q.T, atol=1e-15)))
        self.assertTrue(np.all(np.isclose(np.eye(A.shape[0]), Q.T @ Q, atol=1e-15)))
        self.assertTrue(np.all(np.isclose(A, Q @ R, atol=1e-15)))

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

    def test_random(self):
        np.random.seed(11823568)  # random seed for determinism
        iters = 10
        start = time.time()
        for i in range(iters):
            rows = 100  # np.random.randint(1, 10)
            cols = 100  # np.random.randint(1, 10)
            A = np.random.uniform(-1e10, 1e10, (rows, cols))
            self.runTest(A)
        print(f"Time per iteration: {1000*(time.time()-start)/iters:.2f} ms")


if __name__ == "__main__":
    unittest.main()
