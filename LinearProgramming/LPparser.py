from enum import Enum
import numpy as np

def parse_linear_expression(expression):
    # remove spaces from expression
    expression = "".join(expression.split(" "))

    # parser state declarations
    PARSING_COEFFICIENT = 1
    PARSING_DECISION_VARIABLE = 2

    curr_state = PARSING_COEFFICIENT
    coeffs = []
    decision_vars = []
    coeff = ""
    decision_var = ""
    for char in expression:
        numeric = char.isnumeric()
        negative = (char == '-')
        add = (char == '+')

        # by default, next_state stays as the current state
        next_state = curr_state

        if curr_state == PARSING_COEFFICIENT:
            if not (numeric or add or negative):  # done parsing coefficent, start parsing decision variable
                decision_var += char
                if(coeff == ""):
                    coeff = "1"
                elif(coeff == "-"):
                    coeff = "-1"
                elif(coeff == "+"):
                    coeff = "1"
                coeffs.append(float(coeff))
                coeff = ""  # reset coefficient because it has been completely parsed
                next_state = PARSING_DECISION_VARIABLE
            else:  # not done parsing coefficent
                coeff += char
        elif curr_state == PARSING_DECISION_VARIABLE:
            if(add or negative):  # done parsing decision variable, start parsing coefficent
                coeff += char
                decision_vars.append(decision_var)
                decision_var = ""  # reset decision_var because it has been completely parsed
                next_state = PARSING_COEFFICIENT
            else:
                decision_var += char
        curr_state = next_state

    # should've ended trying to parse a decision variable
    decision_vars.append(decision_var)
    assert len(coeffs) == len(decision_vars)

    return (np.array(coeffs), decision_vars)


def parse_objective(objective):
    min_or_max = "min" if "min" in objective else "max" if "max" in objective else None
    assert(min_or_max in ["min", "max"])
    expression = objective.strip(min_or_max)  # remove min or max from objective
    return (min_or_max, *parse_linear_expression(expression))


def parse_constraint(constraint_expression):
    constraint_expression = "".join(constraint_expression.split(" "))  # remove spaces from constraint_expression

    expression = None
    right_hand_side = None
    constraint_type = None
    for constraint_type in [">=", "<=", "="]:
        if constraint_type in constraint_expression:
            expression, right_hand_side = constraint_expression.split(constraint_type)
            break
    if constraint_type == None:
        raise Exception("Constraint was neither an inequality nor equality")

    (coeffs, decision_vars) = parse_linear_expression(expression)
    right_hand_side = float(right_hand_side)
    return coeffs, decision_vars, constraint_type, right_hand_side


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
    parsed_objective = parse_objective(objective)
    parsed_constraints = [parse_constraint(constraint) for constraint in constraints]
    print(parsed_objective)
    print(parsed_constraints)
    print("done")
