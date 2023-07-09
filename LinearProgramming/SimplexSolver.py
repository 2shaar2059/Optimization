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

    def pivot(self, prev_basic_var_idx, entering_variable):
        """pivot on entering_variable to add it to the basis and replace exiting_variable

        Args:
            prev_basic_var_idx (_type_): index into BASIS variable list
            entering_variable (_type_): index into FULL variable list
        """
        self.basis[prev_basic_var_idx] = entering_variable
        pivot_row = self.matrix[prev_basic_var_idx]
        new_pivot = pivot_row[entering_variable]
        assert 1e-14 < abs(new_pivot)  # TODO what if this fails?

        pivot_row /= new_pivot  # set new pivot location to 1
        pivot = pivot_row[entering_variable]
        assert abs(pivot - 1) < 1e-14

        for row in range(self.matrix.shape[0]):
            if row != prev_basic_var_idx:
                assert row != prev_basic_var_idx
                self.matrix[row] -= (
                    self.matrix[row, entering_variable] / pivot
                ) * pivot_row

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

        found_upper_bound = False
        supremum = 1e99
        basic_var_to_exit = None
        for pivot_to_leave in range(self.num_basic_vars):
            # exiting_var + self.A[pivot_to_leave, entering_variable] * entering =  self.b[pivot_to_leave]
            # exiting_var  =  self.b[pivot_to_leave] - self.A[pivot_to_leave, entering_variable] * entering
            # 0 <= exiting_var
            # 0 <= self.b[pivot_to_leave] - self.A[pivot_to_leave, entering_variable] * entering
            # self.A[pivot_to_leave, entering_variable] * entering <= self.b[pivot_to_leave]
            coeff = self.A[pivot_to_leave, entering_variable]
            if 0 < coeff:
                upper_bound = self.b[pivot_to_leave] / coeff
                if upper_bound < supremum:
                    supremum = upper_bound
                    basic_var_to_exit = pivot_to_leave
                    found_upper_bound = True
        assert found_upper_bound
        return basic_var_to_exit

        """TODO: add logic to detect the objective would not increase
        initial_objective = 0  # TODO
        
        final_objective = 0  # TODO
        assert final_objective <= initial_objective

        """

    def solution(self):
        solution = np.zeros_like(self.c)
        for i, basic_var in enumerate(self.basis):
            solution[basic_var] = self.b[i]
        return solution


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
    np.set_printoptions(linewidth=999)
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
    print(t.basis)
    print(t.matrix)
    while not t.terminated():
        new_basic_var = t.find_variable_to_enter_basis()
        print(f"Entering basis: {new_basic_var}")
        exiting_pivot = t.find_variable_to_exit_basis(new_basic_var)
        print(f"Exiting basis: {t.basis[exiting_pivot]}")
        t.pivot(exiting_pivot, new_basic_var)
        print(t.basis)
        print(t.matrix)
        assert np.all(t.b >= 0)  # BFS should be positive

    print(t.solution())
