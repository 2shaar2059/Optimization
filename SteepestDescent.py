"""This code solves Mx=y for x by iteratively minimizing the associated quadratic form via steepest descent. """

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ====================================================================
# start of the code that is custom to each optimization problem

# M = np.array([[1.0, 2.0],
#               [3.0, 4.0],
#               [5.0, 6.0]])
#
# y = np.array([[7.0],
#               [8.0],
#               [9.0]])

M = np.array([[1.0, 2.0],
              [3.0, 4.0],
              [5.0, 6.0]])

y = np.array([[7.0],
              [8.0],
              [9.0]])



# initial guess
x0 = np.array([[0.0],
               [0.0]])

"""once the norm of the residual falls below this tolerance, stop optimizing"""
RESIDUAL_TOLERANCE = 1e-8

# end of the code that is custom to each optimization problem
# ====================================================================


b = M.T @ y  # TODO REPLACE ME
A = M.T @ M  # TODO REPLACE ME


# The quadratic loss function whose minimum occurs at the same x that minimizes the residual of the original Mx=y
# system in the L2 sense. Evaluating this is not necessary for computing the minimum but rather for seeing into how
# the algorithm works under the hood (curiosity)
def f(x):
    return 0.5 * x.T @ A @ x - b.T @ x


def visualizeLoss(min_x1, max_x1, min_x2, max_x2):
    x1_arr = np.linspace(min_x1, max_x1, 50)
    x2_arr = np.linspace(min_x2, max_x2, 50)

    X, Y = np.meshgrid(x1_arr, x2_arr)
    Z = np.zeros(np.shape(X))
    for i in range(np.shape(Z)[0]):
        for j in range(np.shape(Z)[1]):
            Z[i, j] = f(np.array([[X[i, j], Y[i, j]]]).T)

    ax = plt.figure().gca(projection=Axes3D.name)
    ax.plot_surface(X, Y, Z, cmap='viridis')
    ax.set_title('Loss')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('loss')


def calculate_residual(x):
    return b - A @ x


def SteepestDescent(max_iterations):
    current_x = x0
    residual = calculate_residual(current_x)
    residual_norm = np.linalg.norm(residual)
    iterations = 0

    residual_norm_arr = [residual_norm]
    iteration_arr = [0]

    while iterations < max_iterations and residual_norm > RESIDUAL_TOLERANCE:
        print("Iteration:", iterations)

        gradient = (-residual).T
        step_size = -gradient @ gradient.T / (gradient @ A @ gradient.T)

        print("residual:", residual)
        print()
        current_x += step_size * gradient.T
        residual = calculate_residual(current_x)
        residual_norm = np.linalg.norm(residual)
        iterations += 1

        residual_norm_arr.append(residual_norm)
        iteration_arr.append(iterations)

    plt.plot(iteration_arr, residual_norm_arr)
    plt.xlabel("iteraions")
    plt.ylabel("norm of residual")
    plt.yscale("log")
    plt.grid(True)

    optimized_guess = current_x
    return optimized_guess, iterations


if __name__ == "__main__":
    optimized_guess, iterations_taken = SteepestDescent(10)
    print("Took", iterations_taken, "iterations")
    print("optimized_guess: ", optimized_guess)
    visualizeLoss(-10, 10, -10, 10)
    plt.show()
