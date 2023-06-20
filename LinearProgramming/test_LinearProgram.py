from LinearProgram import *
import unittest
import numpy as np


class Test_LPparser(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        print(str('-'*100))

    def test_parse_constraint(self):
        constraint = Constraint("4x_1 - x_2 + x_4 <= 6")

        coeffs_expected = np.array([4, -1,  1])

        self.assertEqual(len(coeffs_expected), len(constraint.coeffs))
        for i in range(len(coeffs_expected)):
            self.assertAlmostEqual(coeffs_expected[i], constraint.coeffs[i], delta=1e-10)


class Test_Constraint(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        print(str('-'*100))

    def test_convert_LessThan_to_equality_slack(self):
        constraint = Constraint("4x_1 - x_2 + x_4 <= 6")
        constraint.convert_to_equality("slack0")

        coeffs_expected = np.array([4, -1,  1,  1])
        self.assertEqual(len(coeffs_expected), len(constraint.coeffs))
        for i in range(len(coeffs_expected)):
            self.assertAlmostEqual(coeffs_expected[i], constraint.coeffs[i], delta=1e-10)

    def test_convert_GreaterThan_to_equality_slack(self):
        constraint = Constraint("4x_1 - x_2 + x_4 >= 6")
        constraint.convert_to_equality("slack0")

        coeffs_expected = np.array([4, -1,  1,  -1])
        self.assertEqual(len(coeffs_expected), len(constraint.coeffs))
        for i in range(len(coeffs_expected)):
            self.assertAlmostEqual(coeffs_expected[i], constraint.coeffs[i], delta=1e-10)


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
        c_expected = np.array([3,  -2, 1,  0,  -1,  0,  0,  0])

        for i in range(len(lp.A)):
            for j in range(len(lp.A[0])):
                self.assertAlmostEqual(A_expected[i, j], lp.A[i, j], delta=1e-10)

        for i in range(len(lp.b)):
            self.assertAlmostEqual(b_expected[i], lp.b[i], delta=1e-10)

        for i in range(len(lp.c)):
            self.assertAlmostEqual(c_expected[i], lp.c[0, i], delta=1e-10)


if __name__ == "__main__":
    np.set_printoptions(suppress=True)  # don't use scientific notation
    unittest.main()
