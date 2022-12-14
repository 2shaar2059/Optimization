from functools import reduce
import operator

import numpy as np
from LPparser import *


class Objective:
    def __init__(self, objective_expression):
        (self.min_or_max, self.coeffs, self.decision_vars) = parse_objective(objective_expression)


class Constraint:
    def __init__(self, constraint_expression=None, slack_name=None):
        if slack_name:
            self.coeffs = [1.0]
            self.decision_vars = [slack_name]
            self.constraint_type = '>='
            self.right_hand_side = [0.0]

        else:
            (self.coeffs, self.decision_vars, self.constraint_type,
             self.right_hand_side) = parse_constraint(constraint_expression)

    def __is_non_negativity(self):
        return (self.constraint_type == '>=' and
                self.coeffs == [1.0] and
                self.right_hand_side in [0.0, -0.0])

    def __is_equality(self):
        return self.constraint_type == '='

    def needs_slack_variable(self):
        """True if this is an inequality constraint which doesnt represent a non-negativity constraint
        """
        return not self.__is_equality() and not self.__is_non_negativity()

    def __make_greater_than_or_equal(self):
        """Convert from less-than-or-equal-to constraint to greater-than-or-equal-to constraint
        """
        if(self.constraint_type == '<='):  # need to flip constraint
            self.coeffs = [-coeff for coeff in self.coeffs]
            self.right_hand_side *= -1
            self.constraint_type = '>='

    def convert_to_equality(self, slack_var_name):
        """Convert to an equality constraint with a slack variable whose name is given as input parameter

        Args:
            slack_var_name (str): name of slack variable to introduce
        """
        self.__make_greater_than_or_equal()
        if(self.constraint_type == '>='):
            self.decision_vars.append(slack_var_name)
            self.coeffs.append(-1.0)
            self.constraint_type = '='


class LinearProgram:
    def __init__(self, objective, constraints):
        self.objective: Objective = Objective(objective)
        self.constraints: list[Constraint] = [Constraint(constraint) for constraint in constraints]

        objective_vars = set(self.objective.decision_vars)
        constraint_vars = set().union(*[constraint.decision_vars for constraint in self.constraints])

        extraneous_vars = constraint_vars - objective_vars
        if extraneous_vars != set():
            raise Exception(
                f"{extraneous_vars} found in constraints but not in objective")

        # ensure decision variable naming con't match with slack and artifical variable naming
        # for var in objective_vars:
        #     assert(
        #         'slack_' not in var), f"Decision variable {var} cannot start with \"slack_\" since that naming is reserved for slack variables"
        #     assert(
        #         'artif_' not in var), f"Decision variable {var} cannot start with \"artif_\" since that naming is reserved for artificial variables"

        unconstrained_vars = objective_vars - constraint_vars
        if unconstrained_vars != set():
            print(f"Found Unconstrianed Variables: {unconstrained_vars}")

        slack_var_constraints = []
        slack_var_count = 0
        for constraint in self.constraints:
            if constraint.needs_slack_variable():
                new_slack_var = f"slack_{slack_var_count}"
                constraint.convert_to_equality(new_slack_var)
                slack_var_constraints.append(Constraint(slack_name=new_slack_var))
                slack_var_count += 1
        self.constraints += slack_var_constraints
        print("F")
        # for unconstrained_var in unconstrained_vars:
        # introduce artifical variable


"""
inequality constriants: ax >= b
ax >= b
ax - b >= 0
ax - b = x'

x' >= 0
ax - x' = b


inequality constriants: ax <= b
-ax >= -b

x' >= 0
-ax - x' = -b
"""


if __name__ == "__main__":
    # objective = "max 3x_1 - 2x_2 - x_3 + x_4"
    objective = "max 3x_1 - 2x_2 - x_3 + x_4 + arti"
    constraints = [
        "4x_1 - x_2 + x_4 <= 6",
        "-7x_1 + 8x_2 + x_3 >= 7",
        "x_1 + x_2 + 4x_4 = 12",
        "x_1  >= 0",
        "x_2  >= 0",
        "x_3  >= 0",

        # "_xsdf  >= 0",
        # "sdfx  >= 0",
    ]

    lpp = LinearProgram(objective, constraints)
