import numpy as np
import unittest
from ClassicGramSchmidt import ClassicGramSchmidt


class Test_ClassicGramSchmidt(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        print(str("-" * 100))

    def test_ComplexTranslationOnly(self):
        A = np.array(
            [
                [1, 2, 3],
                [1, 3, 3],
                [2, 3, 4],
            ]
        ).astype(np.float64)

        Q, R = ClassicGramSchmidt(A)

        self.assertTrue(np.all(np.isclose(np.eye(A.shape[0]), Q @ Q.T, atol=1e-6)))
        self.assertTrue(np.all(np.isclose(np.eye(A.shape[0]), Q.T @ Q, atol=1e-6)))
        self.assertTrue(np.all(np.isclose(A, Q @ R, atol=1e-6)))


if __name__ == "__main__":
    unittest.main()
