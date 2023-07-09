import numpy as np
from rref import RREF

"""NOTES:

    basic variables >= 0
    non-basic variabels == 0
"""


class Tableau:
    def __init__(self, A, b, c):
        self.matrix = np.concatenate(
            (
                np.concatenate((A, b), axis=1),
                np.concatenate((c.T, [[0]]), axis=1),
            ),
            axis=0,
        )

    @property
    def A(self):
        return self.matrix[:-1, :-1]

    @property
    def b(self):
        return self.matrix[:-1, -1]

    @property
    def c(self):
        return self.matrix[-1, :-1]

    def reduce_BFS(self, basis):
        self.matrix = RREF(self.matrix, basis)[0]

    def pivot(self, prev_basic_var_idx, entering_variable):
        """pivot on entering_variable to add it to the basis and replace exiting_variable

        Args:
            prev_basic_var_idx (_type_): index into BASIS variable list
            entering_variable (_type_): index into FULL variable list
        """
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

    def __str__(self):
        return f"{self.matrix}"


class SimplexSolver:
    def __init__(self, A, b, c):
        num_constraints, num_variables = A.shape
        self.c_original = c
        self.num_basic_vars = num_constraints
        self.num_nonbasic_vars = num_variables - num_constraints
        self.tableau = Tableau(A, b, c)

    @property
    def A(self):
        return self.tableau.A

    @property
    def b(self):
        return self.tableau.b

    @property
    def c(self):
        return self.tableau.c

    def compute_initial_BFS(self, initial_basis=None):
        # TODO initialize basis by solving auxillary LP
        self.basis = initial_basis or list(range(self.num_basic_vars))
        self.tableau.reduce_BFS(self.basis)
        assert np.all(self.b >= 0)  # BFS should be positive

    def unbounded(self):
        return np.all(self.c <= 0)

    def infeasible(self):
        return True  # TODO fixme

    def terminated(self):
        return np.all(self.c >= 0)

    def find_new_basis_var(self):
        assert not self.terminated()
        return np.argmin(self.c)

    def add_new_basis_var(self, entering_variable):
        # TODO : find which basic variable becomes 0 after pushing entering_variable to minimize objective
        assert np.all(self.b >= 0)  # BFS should be positive

        found_upper_bound = False
        supremum = 1e99
        basic_var_to_exit = None
        for pivot_to_leave in range(self.num_basic_vars):
            coeff = self.A[pivot_to_leave, entering_variable]
            if 0 < coeff:
                if self.b[pivot_to_leave] == 0.0:
                    print(f"Degeneracy detected: b[{pivot_to_leave}] = 0")
                    return None  # TODO handle this case and don't return
                upper_bound = self.b[pivot_to_leave] / coeff
                if upper_bound < supremum:
                    supremum = upper_bound
                    basic_var_to_exit = pivot_to_leave
                    found_upper_bound = True
        if not found_upper_bound:
            print(f"problem unbounded. Make variable {entering_variable} = +inf")
            return None
        return basic_var_to_exit

    def solution(self):
        solution = np.zeros_like(self.c)
        for i, basic_var in enumerate(self.basis):
            solution[basic_var] = self.b[i]
        return solution

    def objective(self):
        return np.dot(self.c_original.T, self.solution())[0]

    def solve(self):
        self.compute_initial_BFS([1, 2])
        prev_objective = self.objective()

        print(self.tableau)
        print(f"Basis: {self.basis}")
        print(f"BFS: {self.solution()}")
        print(f"initial Objective: {prev_objective}")
        print()

        i = 0
        while not self.terminated():
            new_basic_var = self.find_new_basis_var()
            print(f"Entering basis: {new_basic_var}")

            exiting_pivot = self.add_new_basis_var(new_basic_var)
            if exiting_pivot is None:  # early termination
                break
            print(f"Exiting basis: {self.basis[exiting_pivot]}")

            self.basis[exiting_pivot] = new_basic_var
            self.tableau.pivot(exiting_pivot, new_basic_var)
            print(f"Basis: {self.basis}")
            print(self.tableau)
            assert np.all(self.b >= 0)  # BFS should be positive

            curr_objective = self.objective()
            print(f"Iter {i} Objective: {curr_objective}")

            assert curr_objective <= prev_objective

            print(f"BFS: {self.solution()}")
            print()

            prev_objective = curr_objective
            i += 1
        return self.solution()


if __name__ == "__main__":
    np.set_printoptions(linewidth=999)

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
    ss.solve()
