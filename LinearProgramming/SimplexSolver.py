from LinearProgram import LinearProgram
import numpy as np

"""NOTES:

    basic variables >= 0
    non-basic variabels == 0
"""


class Tableau:
    def __init__(self, A, b, c):
        num_constraints, num_variables = A.shape
        self.num_basic_vars = num_constraints
        self.num_nonbasic_vars = num_variables - num_constraints
        self.matrix = np.concatenate(
            (
                np.concatenate((A, b), axis=1),
                np.concatenate((c.T, [[0]]), axis=1),
            ),
            axis=0,
        )

    @property
    def c(self):
        return self.tableau[-1, :-1]

    def unbounded(self):
        return np.all(self.c <= 0)

    def infeasible(self):
        return True  # TODO fixme

    def terminated(self):
        return np.all(self.c >= 0)

    def pivot(self, idxs):
        """pivot to try to decrease objective

        Args:
            idxs (_type_): _description_

        Returns:
            _type_: _description_
        """
        initial_objective = TODO

        final_objective = TODO

        assert final_objective <= initial_objective


# class SimplexSolver:
#     def initialize():
#         bfs = np.zeros_like(self.c)  # TODO initalize by solving the auxillary LP
#         tableau = TODO

#         return tableau


#     def solve(lp):
#         tableau = initialize(lp)
#         while not terminated(tableau):
#             tableau = pivot(tableau)
#         return tableau.solution()


if __name__ == "__main__":
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
    t = Tableau(A, b, c)
    print(t.matrix)
