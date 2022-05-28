"""This code finds the roots of a complicated function by performing the Newton-Raphson iteration process on a range
of x values across a given domain of the function and seeing which values they converge to. """

import numpy as np
from matplotlib import pyplot as plt

# ====================================================================
# start of the code that is custom to each optimization problem

# defining the domain over which to search for roots
X_MIN = -10
X_MAX = 10

NUM_GUESSES = 1000  # number of points on domain on which to perform the Newton Raphson iteration scheme


def f(x):  # The function whose roots are to be found
    return 1 + 10 * (x * 3 % 2) / (x * x + 1) * np.sin(2 * x) + np.sin(x) * np.sqrt(np.abs(x)) - np.cos(4 * x)


F_TOLERANCE = 1e-10  # if f(x) is within F_TOLERANCE of 0, x is considered a root

"""A group of different x values clustered around a root should be considered a single root if they are within 
X_TOLERANCE of one another."""
X_TOLERANCE = 1e-8


# end of the code that is custom to each optimization problem
# ====================================================================

def isRoot(purported_root):
    return abs(f(purported_root)) < F_TOLERANCE


def getUniqueRoots(roots):
    """

    :param roots: list of roots
    :return: list of unique roots (if roots in the input array are within X_TOLERANCE of
    each-other, only one of them is stored in the unique roots array)
    """
    roots.sort()
    unique_roots = []
    prev_root = None
    for root in roots:
        if prev_root is None or abs(root - prev_root) > X_TOLERANCE:
            unique_roots.append(root)
            prev_root = root
    return unique_roots


def graph_f():
    """ graph the function f by evaluating 1,000,000 points on the domain """
    x = np.linspace(X_MIN, X_MAX, int(1e6))
    y = np.apply_along_axis(f, axis=0, arr=x)
    plt.plot(x, y, 'b-', label="f(x)")
    plt.legend()
    plt.grid(True)


def NewtonRaphson(initial_guess, max_iterations):
    dx = 1e-10
    current_x = initial_guess
    iterations = 0
    while iterations < max_iterations and abs(f(current_x)) > F_TOLERANCE:
        slope = (f(current_x + dx) - f(current_x)) / dx
        if not slope == 0:
            current_x += -f(current_x) / slope
        iterations += 1
    optimized_guess = current_x
    return optimized_guess, iterations


if __name__ == "__main__":
    initial_guesses = np.linspace(X_MIN, X_MAX, NUM_GUESSES)
    optimized_guesses = [NewtonRaphson(initial_guess, 100)[0] for initial_guess in initial_guesses]

    # making sure the optimized guesses are actually roots and didn't just diverge to a non-root from the initial guess
    roots = getUniqueRoots([optimized_guess for optimized_guess in optimized_guesses if isRoot(optimized_guess)])

    print("Roots of f occur at following list of x values:", roots)

    graph_f()
    plt.plot(roots, [f(root) for root in roots], "g*", label="roots")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.xlim(X_MIN, X_MAX)
    plt.legend()
    plt.grid(True)
    plt.show()
