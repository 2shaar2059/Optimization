from LinearProgram import LinearProgram

class Tableau:
    def __init__(self, lp):
        self.c = lp.c

        self.num_nonbasic_vars = len(lp.constraints)
        self.num_basic_vars = lp.
        self.


def initialize():
    tableau = TODO

    return tableau


def pivot():
    initial_objective = TODO

    final_objective = TODO

    assert final_objective >= initial_objective
    return tableau


def unbounded(tableau):
    return TODO


def infeasible(tableau):
    return TODO


def terminated(tableau):
    return TODO


def solve(lp):
    tableau  = initialize(lp)
    while (not terminated(tableau)):
        tableau = pivot(tableau)
    return tableau.solution()