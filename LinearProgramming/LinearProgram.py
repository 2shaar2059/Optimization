from ordered_set import OrderedSet
import numpy as np
from LPparser import parse_constraint, parse_objective


class Objective:
    def __init__(self, objective_expression):
        (self.min_or_max, self.coeffs, self.decision_vars) = parse_objective(
            objective_expression
        )

    def standardize(self):
        if self.min_or_max == "min":
            self.min_or_max == "max"
            self.coeffs *= -1


class Constraint:
    def __init__(self, constraint_expression=None, slack_name=None):
        if slack_name:
            self.coeffs = [1.0]
            self.decision_vars = [slack_name]
            self.constraint_type = ">="
            self.right_hand_side = 0.0
        else:
            (
                self.coeffs,
                self.decision_vars,
                self.constraint_type,
                self.right_hand_side,
            ) = parse_constraint(constraint_expression)

    def __is_non_negativity(self):
        return (
            self.constraint_type == ">="
            and (len(self.coeffs) == 1 and self.coeffs[0] == 1.0)
            and self.right_hand_side in [0.0, -0.0]
        )

    def __is_equality(self):
        return self.constraint_type == "="

    def needs_slack_variable(self):
        """True if this is an inequality constraint which doesnt represent a non-negativity constraint"""
        return not self.__is_equality() and not self.__is_non_negativity()

    def convert_to_equality(self, slack_var_name):
        """Convert to an equality constraint with a slack variable whose name is given as input parameter

        Args:
            slack_var_name (str): name of slack variable to introduce
        """
        if self.constraint_type == ">=":
            self.decision_vars.append(slack_var_name)
            self.coeffs = np.append(self.coeffs, [-1.0])
            self.constraint_type = "="
        elif self.constraint_type == "<=":
            self.decision_vars.append(slack_var_name)
            self.coeffs = np.append(self.coeffs, [1.0])
            self.constraint_type = "="

    def __str__(self):
        expression = "+ ".join(
            f"{coeff}{var} " for coeff, var in zip(self.coeffs, self.decision_vars)
        )
        expression += f"{self.constraint_type} {self.right_hand_side}"
        return expression


class LinearProgram:
    # TODO stop iterating over self.constriant so many times (slow?)
    def __init__(self, objective, constraints):
        """Create a LP in standard form (min c^Tx s.t. Ax = b, x>=0)

        Args:
            objective (_type_): _description_
            constraints (_type_): _description_
        """
        self.objective = Objective(objective)
        self.objective.standardize()

        self.constraints = [Constraint(constraint) for constraint in constraints]

        self.__check_extraneous_vars()
        self.__create_slack_vars()

        # now, all constraints should be equality or non-negativity constraints
        self.__create_artificial_vars()
        self.__create_A_b_c()

    def __check_extraneous_vars(self):
        objective_vars = set(self.objective.decision_vars)
        constraint_vars = set().union(
            *[constraint.decision_vars for constraint in self.constraints]
        )

        non_descion_vars = constraint_vars - objective_vars
        if non_descion_vars != set():
            print(f"{non_descion_vars} appear in constraints but in objective")

        unconstrained_vars = objective_vars - constraint_vars
        if unconstrained_vars != set():
            print(f"{unconstrained_vars} unconstrained")

        # ensure decision variable naming con't match with slack and artifical variable naming
        for var in objective_vars:
            assert (
                "slack" not in var
            ), f'Decision variable {var} cannot start with "slack_"; that naming is reserved for slack variables'
            assert (
                "artif" not in var
            ), f'Decision variable {var} cannot start with "artif_"; that naming is reserved for artificial variables'

    def __create_slack_vars(self):
        """TODO optimize: dont add slack variable for simple relations like x >= 3
        instead add (x=x'+3) to a table of special constraints that dont make it to the final A matrix
        and add x' >= 0 as a slack variable constraint)
        """
        # Add slack variables to turn ineqaulity constraints into equality constraints
        slack_var_count = 0
        for constraint in self.constraints:
            if constraint.needs_slack_variable():
                new_slack_var = f"slack_{slack_var_count}"
                constraint.convert_to_equality(new_slack_var)
                self.constraints.append(Constraint(slack_name=new_slack_var))
                slack_var_count += 1

    def __create_artificial_vars(self):
        # identifying the unconstrained variables as those appearing in equality but not non-negativity constraints
        equality_vars, non_negativity_vars = set(), set()
        for constraint in self.constraints:
            if constraint.constraint_type == "=":
                equality_vars.update(constraint.decision_vars)
            else:
                non_negativity_vars.update(constraint.decision_vars)
        unconstrained_vars = equality_vars - non_negativity_vars

        # Add artifical variables to represent unconstrained variables
        artifical_base = "base"
        if unconstrained_vars:
            self.constraints.append(Constraint(slack_name=artifical_base))
        for unconstrained_var in unconstrained_vars:
            artifical_var = unconstrained_var + "_artif"
            # adding artifical constraint: unconstrained_var = artifical_var - artifical_base
            self.constraints.append(
                Constraint(f"{unconstrained_var}-{artifical_var}+{artifical_base}=0")
            )
            self.constraints.append(Constraint(slack_name=artifical_var))

        """TODO optimize: substitute artifical equality constriant into preexsiting equlaity constrinats and objective
        instead add (x=x'-x'') to a table of special constraints,
        replace all instances of x in objective and original constraints with (x'-x'') (and appropriatly distribute coefficients of x)
        and add x', x'' >= 0 as a slack variable constraint)
        """

    def __create_A_b_c(self):
        # finding all the variables (decsion_vars, slack_Vars, artificila_vars) to optimize over
        x = OrderedSet()
        for constraint in self.constraints:
            if constraint.constraint_type == "=":
                x.update(constraint.decision_vars)

        A_cols = len(x)
        A_rows = len([c for c in self.constraints if c.constraint_type == "="])

        # finally creating the A,B,c matrices
        self.A = np.zeros((A_rows, A_cols))
        self.b = np.zeros((A_rows, 1))
        A_row_idx = 0
        for constraint in self.constraints:
            if constraint.constraint_type == "=":
                self.b[A_row_idx] = constraint.right_hand_side
                for coeff, var in zip(constraint.coeffs, constraint.decision_vars):
                    A_col_idx = x.index(var)
                    self.A[A_row_idx, A_col_idx] = coeff
                A_row_idx += 1

        self.c = np.zeros((1, A_cols))
        for coeff, var in zip(self.objective.coeffs, self.objective.decision_vars):
            c_col_idx = x.index(var)
            self.c[0, c_col_idx] = coeff

        self.x = x.items


if __name__ == "__main__":
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

    print(f"Problem type:\n{lp.objective.min_or_max}")
    print(f"c^T:{lp.c}")
    print(f"A:\n{lp.A}")
    print(f"x: {lp.x}")
    print(f"b:\n{lp.b}")
