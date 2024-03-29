from SimplexSolver import SimplexSolver
import unittest
import numpy as np


class Test_SimplexSolver(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        print(str("-" * 100))

    def checkSolution(self, estimated, expected):
        tolerance = 1e-14
        self.assertLessEqual(abs(estimated - expected).max(), tolerance)

    def test_1(self):
        # Example from wenshenpsu Linear Programming YouTube Series
        A = np.array(
            [
                [-6, 0, 1, -2, 2],
                [3, 1, -1, 8, 1],
            ]
        )
        b = np.array([[6, 9]]).T
        c = np.array(
            [
                [-4, 1, 1, 7, 3],
            ]
        ).T
        ss = SimplexSolver(A, b, c)
        solution = ss.solve()

        expected_solution = np.array([1, 0, 0, 0, 6])
        self.checkSolution(solution, expected_solution)

    def test_2(self):
        # Example from wenshenpsu Linear Programming YouTube Series
        A = np.array(
            [
                [3, 2, 0, 1, 0, 0],
                [-1, 1, 4, 0, 1, 0],
                [2, -2, 5, 0, 0, 1],
            ]
        )
        b = np.array([[60, 10, 50]]).T
        c = np.array(
            [
                [-2, -3, -3, 0, 0, 0],
            ]
        ).T
        ss = SimplexSolver(A, b, c)
        solution = ss.solve()

        expected_solution = np.array([8, 18, 0, 0, 0, 70])
        self.checkSolution(solution, expected_solution)

    def test_degeneracy_ok(self):
        # Example from wenshenpsu Linear Programming YouTube Series
        A = np.array(
            [
                [8, -2, 1, -1, 1, 0, 0],
                [2, 5, 0, 2, 0, 1, 0],
                [1, -1, 2, -4, 0, 0, 1],
            ]
        )
        b = np.array([[50, 150, 100]]).T
        c = np.array(
            [
                [2, 4, -4, 7, 0, 0, 0],
            ]
        ).T
        ss = SimplexSolver(A, b, c)
        solution = ss.solve([4, 5, 6])

        expected_solution = np.array([0, 0, 50, 0, 0, 150, 0])
        self.checkSolution(solution, expected_solution)

    def test_redundant_constraint(self):
        # Example from wenshenpsu Linear Programming YouTube Series
        A = np.array(
            [
                [1, 2, 0, 1],
                [2, 1, 1, 0],
                [-1, 4, -2, 3],
            ]
        )
        b = np.array([[20, 10, 40]]).T
        c = np.array(
            [
                [1, 4, 3, 2],
            ]
        ).T
        ss = SimplexSolver(A, b, c)
        solution = ss.solve()

        expected_solution = np.array([1, 0, 0, 6])
        self.checkSolution(solution, expected_solution)


if __name__ == "__main__":
    np.set_printoptions(linewidth=999, suppress=True)
    unittest.main()
