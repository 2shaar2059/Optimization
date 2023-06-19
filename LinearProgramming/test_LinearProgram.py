from LinearProgram import *
import unittest
import numpy as np


class Test_LinearProgram(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        print(str('-'*100))

    def test_Parsing1(self):
        objective = "max 3x_1 - 2x_2 - x_3 + x_4"
        constraints = [
            "4x_1 - x_2 + x_4 <= 6",
            "-7x_1 + 8x_2 + x_3 >= 7",
            "x_1 + x_2 + 4x_4 = 12",
            "x_1  >= 0",
            "x_2  >= 0",
            "x_3  >= 0",
        ]
        lp = LinearProgram(objective, constraints)

        A_expected = np.array([
            [4, -1,  1,  1,  0,  0,  0,  0],
            [-7,  8,  0,  0,  1, -1,  0,  0],
            [1,  1,  4,  0,  0,  0,  0,  0],
            [0,  0,  1,  0,  0,  0, -1,  1,]])
        b_expected = np.array([[6, 7, 12, 0]]).T
        c_expected = np.array([[-3,  2, -1,  0,  1,  0,  0,  0]])

        x: ['x_1', 'x_2', 'x_4', 'slack_0', 'x_3', 'slack_1', 'x_4_artif', 'base']

        self.assertLess(np.abs(A_expected - lp.A).max(), 1e-10)
        self.assertLess(np.abs(b_expected - lp.b).max(), 1e-10)
        self.assertLess(np.abs(c_expected - lp.c).max(), 1e-10)


if __name__ == "__main__":
    np.set_printoptions(suppress=True)  # don't use scientific notation
    unittest.main()
