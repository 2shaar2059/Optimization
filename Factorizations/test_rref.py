import time
import numpy as np
import unittest
from rref import RREF


class Test_RREF(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        print(str("-" * 100))

    def runTest(self, A):
        rref, E = RREF(A.astype(np.float64))
        tolerance = 1e-11
        max_tolerable_error = tolerance * abs(A).max()
        max_error = abs(E @ A - rref).max()
        if max_tolerable_error < max_error:
            print(A)
        self.assertLessEqual(max_error, max_tolerable_error)

    def test_skinny(self):
        A = np.array(
            [
                [3, 2, 3],
                [6, 3, 3],
                [2, 3, 4],
                [5, 3, 4],
            ]
        )
        self.runTest(A)

    def test_wide(self):
        A = np.array(
            [
                [3, 2, 3, 4],
                [6, 3, 3, 6],
                [2, 3, 4, 1],
            ]
        )
        self.runTest(A)

    def test_square(self):
        A = np.array(
            [
                [3, 2, 3],
                [2, 3, 4],
                [5, 3, 4],
            ]
        )
        self.runTest(A)

    def test_poorlyConditioned(self):
        # testcase taken from: https://fgiesen.wordpress.com/2013/06/02/modified-gram-schmidt-orthogonalization/
        epsilon = 1e-15
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
            rows = np.random.randint(1, 100)
            cols = np.random.randint(1, 100)
            A = np.random.uniform(-1e5, 1e5, (rows, cols))
            self.runTest(A)
        print(f"Time per iteration: {1000*(time.time()-start)/iters:.2f} ms")

    def test_randomEdgecases(self):  # all matrix entries are 0 or 1
        np.random.seed(11823568)  # random seed for determinism
        iters = 10000
        start = time.time()
        for i in range(iters):
            rows = 3
            cols = 3
            A = np.random.randint(0, 2, (rows, cols))
            self.runTest(A)
        print(f"Time per iteration: {1000*(time.time()-start)/iters:.2f} ms")

    def test_randomint_2x2(self):
        np.random.seed(11823568)  # random seed for determinism
        iters = 10000
        start = time.time()
        for i in range(iters):
            A = np.random.randint(-10, 10, (2, 2))
            self.runTest(A)
        print(f"Time per iteration: {1000*(time.time()-start)/iters:.2f} ms")

    def test_randomint_3x3(self):
        np.random.seed(11823568)  # random seed for determinism
        iters = 1000
        start = time.time()
        for i in range(iters):
            A = np.random.randint(-10, 10, (3, 3))
            self.runTest(A)
        print(f"Time per iteration: {1000*(time.time()-start)/iters:.2f} ms")


if __name__ == "__main__":
    np.set_printoptions(suppress=True, linewidth=999999)
    unittest.main()
