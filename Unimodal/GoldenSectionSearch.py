import numpy as np
from matplotlib import pyplot as plt
from math import sin, cos, sqrt


NUM_FUNCTION_EVALS = 0


def f(x):
    global NUM_FUNCTION_EVALS
    NUM_FUNCTION_EVALS += 1
    return (x-3.5)**2


def plot(a, b):
    x_values = np.linspace(a, b, 500)
    f_values = [f(x) for x in x_values]
    plt.plot(x_values, f_values)


phi = 0.5*(1+sqrt(5.0))  # golden ratio


def min_search(min_bound, max_bound, iterations_left):
    assert (max_bound > min_bound)

    # plot current best guess and lebel wth iteration number
    current_best_guess = 0.5*(max_bound+min_bound)
    current_best_guess_result = f(current_best_guess)
    plt.scatter(current_best_guess, current_best_guess_result)

    if (iterations_left == 0):
        plt.text(current_best_guess, current_best_guess_result, "OPTIMUM")
        return current_best_guess
    else:
        plt.text(current_best_guess, current_best_guess_result, str(iterations_left))

    full_range = max_bound-min_bound

    """
    finding smaller_range and larger_range ranges which sum to full_range such that larger_range = phi * smaller_range
    """
    smaller_range = full_range/(1+phi)
    larger_range = phi*smaller_range

    assert (abs(full_range-smaller_range-larger_range) < 1e-9*full_range)
    assert (abs(phi - larger_range/smaller_range) < 1e-9)

    left_evaluation_point = min_bound+smaller_range
    right_evaluation_point = min_bound+larger_range

    a = f(min_bound)
    b = f(left_evaluation_point)
    c = f(right_evaluation_point)
    d = f(max_bound)
    if (a >= b and b <= c):
        # ignore the interval [right_evaluation_point, max_bound]
        return min_search(min_bound, right_evaluation_point, iterations_left-1)
    elif (b >= c and c <= d):
        # ignore the interval [min_bound, left_evaluation_point]
        return min_search(left_evaluation_point, max_bound, iterations_left-1)
    elif (a < b):
        # ignore the interval [left_evaluation_point, max_bound]
        return min_search(min_bound, left_evaluation_point, iterations_left-1)
    elif (c > d):
        # ignore the interval [min_bound, right_evaluation_point]
        return min_search(right_evaluation_point, max_bound, iterations_left-1)
    else:
        print("uh oh didn't expect to get here")
        print(min_bound, left_evaluation_point, right_evaluation_point, max_bound)
        print(a, b, c, d)


if __name__ == "__main__":
    plot(2, 4)
    NUM_FUNCTION_EVALS = 0
    x_min = min_search(2, 4, 5)
    print(x_min, f(x_min))
    print("NUM_FUNCTION_EVALS: ", NUM_FUNCTION_EVALS)
    plt.show()
