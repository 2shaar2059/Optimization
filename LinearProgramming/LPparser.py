from enum import Enum
import numpy as np


def parse_linear_expression(expression):
    #remove spaces from expression 
    expression = "".join(expression.split(" "))


    #parser state declarations
    PARSING_COEFFICIENT = 1
    PARSING_DECSION_VARIABLE = 2

    curr_state = PARSING_COEFFICIENT
    coeffs = []
    descion_vars = []
    coeff = ""
    descion_var = ""
    for char in expression:
        numeric = char.isnumeric()
        negative = (char == '-')
        add = (char == '+')

        # by default, next_state stays as the current state
        next_state = curr_state

        if curr_state == PARSING_COEFFICIENT:
            if not (numeric or add or negative): # done parsing coefficent, start parsing descion variable
                descion_var += char
                if(coeff == ""): coeff = "1"
                elif(coeff == "-"): coeff = "-1"
                elif(coeff == "+"): coeff = "1"
                coeffs.append(float(coeff))
                coeff = "" #reset coefficient because it has been completely parsed
                next_state = PARSING_DECSION_VARIABLE
            else: #not done parsing coefficent
                coeff+=char
        elif curr_state == PARSING_DECSION_VARIABLE:
            if(add or negative): # done parsing descion variable, start parsing coefficent
                coeff += char
                descion_vars.append(descion_var)
                descion_var = "" #reset descion_var because it has been completely parsed
                next_state = PARSING_COEFFICIENT
            else:
                descion_var += char
        curr_state = next_state

    #should've ended trying to parse a decsion variable
    descion_vars.append(descion_var)    
    assert len(coeffs) == len(descion_vars)
    
    return (coeffs, descion_vars)


def parse_problem(problem):
    #remove spaces from problem 
    problem = "".join(problem.split(" "))
    
    min_or_max = None
    min_or_max = "min" if "min" in problem else "max" if "max" in problem else None
    assert(min_or_max in ["min", "max"])

    #remove min or max from problem
    expression = problem.strip(min_or_max)
    assert(min_or_max not in ["min" or "max"])

    return (min_or_max, *parse_linear_expression(expression))


def parse_constraints(constraints):
    constraint_types = [">=", "<=", "="]
    
    parsed_constaints = []
    for constraint in constraints:
        #remove spaces from constraint 
        constraint = "".join(constraint.split(" "))

        expression = None
        right_hand_side = None
        constraint_type = None
        for constraint_type in constraint_types: 
            if constraint_type in constraint:
                expression, right_hand_side = constraint.split(constraint_type)
                break
        if constraint_type == None:
            raise Exception("Constraint was neither an inequality nor equality") 
        
        (coeffs, descion_vars) = parse_linear_expression(expression) 
        right_hand_side = float(right_hand_side)
        parsed_constaints.append((coeffs, descion_vars, constraint_type, right_hand_side))
    return parsed_constaints

class LPparser:
    def __init__(self, problem, constraints):
        self.problem = parse_problem(problem)
        self.constraints = parse_constraints(constraints)
        # self.decsion_vars = []
        # self.A = None
        # self.b = None
        # self.c = None


if __name__ == "__main__":
    problem = "max 3x_1 - 2x_2 - x_3 + x_4"
    constraints = [
        "4x_1 - x_2 + x_4 <= 6",
        "-7x_1 + 8x_2 + x_3 >= 7",
        "x_1 + x_2 + 4x_4 = 12",
        "x_1  >= 0",
        "x_2  >= 0",
        "x_3  >= 0",
    ]

    lpp = LPparser(problem,constraints)
