from LinearProgram import LinearProgram
import numpy as np
from rref import RREF

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

    def compute_initial_BFS(self, initial_basis=None):
        # TODO initialize basis by solving auxillary LP
        self.basis = initial_basis or list(range(self.num_basic_vars))
        self.matrix = RREF(self.matrix, self.basis)[0]

        assert np.all(self.b >= 0)  # BFS should be positive

    @property
    def A(self):
        return self.matrix[:-1, :-1]

    @property
    def b(self):
        return self.matrix[:-1, -1]

    @property
    def c(self):
        return self.matrix[-1, :-1]

    def unbounded(self):
        return np.all(self.c <= 0)

    def infeasible(self):
        return True  # TODO fixme

    def terminated(self):
        return np.all(self.c >= 0)

    def pivot(self, exiting_variable, entering_variable):
        """pivot on entering_variable to add it to the basis and replace exiting_variable

        Args:
            exiting_variable (int): index into

        Returns:
            _type_: _description_
        """

        assert np.all(self.b >= 0)  # BFS should be positive

    def find_variable_to_enter_basis(self):
        """_summary_

        Returns:
            int: idx of variable with
        """
        assert not self.terminated()
        return np.argmin(self.c)

    def find_variable_to_exit_basis(self, entering_variable):
        # TODO : find which basic variable becomes 0 after pushing entering_variable to minimize objective
        assert np.all(self.b >= 0)  # BFS should be positive

        supremum = 1e99
        basic_var_to_exit = None
        for pivot_row_idx, basic_var in enumerate(self.basis):
            upper_bound = (
                self.b[pivot_row_idx] / self.A[pivot_row_idx, entering_variable]
            )
            assert 0 < upper_bound  # TODO what if upper bound is negative?
            if upper_bound < supremum:
                supremum = upper_bound
                basic_var_to_exit = basic_var
        assert basic_var_to_exit is not None
        return basic_var_to_exit
        """TODO: add logic to detect the objective would not increase
        initial_objective = 0  # TODO
        
        final_objective = 0  # TODO
        assert final_objective <= initial_objective

        """


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
    t.compute_initial_BFS([2, 1])
    print(t.matrix)
    new_basic_var = t.find_variable_to_enter_basis()
    print(f"Entering basis: {new_basic_var}")
    var_to_exit_basis = t.find_variable_to_exit_basis(new_basic_var)
    print(f"Exiting basis: {var_to_exit_basis}")
