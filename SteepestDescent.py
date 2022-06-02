"""This code solves Mx=y for x by iteratively minimizing the associated quadratic form via steepest descent. """

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ====================================================================
# start of the code that is custom to each optimization problem

M = np.array([[0.1, 0.2],
              [0.3, 0.4],
              [0.5, 0.6]])

y = np.array([[0.7],
              [0.8],
              [0.1]])

# initial guess
x0 = np.array([[2.0],
               [4.0]])

"""once the norm of the residual falls below this tolerance, stop optimizing"""
RESIDUAL_TOLERANCE = 1e-10

MAX_ITERATIONS = 100

# end of the code that is custom to each optimization problem
# ====================================================================


b = M.T @ y
A = M.T @ M


# The quadratic loss function whose minimum occurs at the same x that minimizes the residual of the original Mx=y
# system in the L2 sense. Evaluating this is not necessary for computing the minimum but rather for seeing into how
# the algorithm works under the hood (curiosity)
def f(x):
    # return (0.5 * x.T @ A @ x - b.T @ x)[0][0]
    # return np.linalg.norm(M @ x - y, 2)
    y_residual = M @ x - y
    return (y_residual.T @ y_residual)[0][0]


def visualize_f(min_x1, max_x1, min_x2, max_x2):
    x1_arr = np.linspace(min_x1, max_x1, 20)
    x2_arr = np.linspace(min_x2, max_x2, 20)

    X, Y = np.meshgrid(x1_arr, x2_arr)
    Z = np.zeros(np.shape(X))
    for i in range(np.shape(Z)[0]):
        for j in range(np.shape(Z)[1]):
            Z[i, j] = f(np.array([[X[i, j], Y[i, j]]]).T)

    ax = plt.figure().gca(projection=Axes3D.name)
    ax.plot_surface(X, Y, Z, cmap='viridis')
    ax.set_title('f(x)')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('f')


def SteepestDescent():
    current_x = x0
    residual = b - A @ current_x
    residual_norm = np.linalg.norm(residual)
    iterations = 0

    b_residual_norm_arr = [residual_norm]
    f_arr = [f(current_x)]
    iteration_arr = [0]

    while iterations < MAX_ITERATIONS and residual_norm > RESIDUAL_TOLERANCE:
        print("Iteration:", iterations)

        gradient = (-residual).T
        step_size = -gradient @ gradient.T / (gradient @ A @ gradient.T)

        print("residual:", residual)
        print()

        current_x += step_size * gradient.T
        residual = b - A @ current_x
        residual_norm = np.linalg.norm(residual, 2)
        iterations += 1

        b_residual_norm_arr.append(residual_norm)
        f_arr.append(f(current_x))
        iteration_arr.append(iterations)

    plt.subplot(2, 1, 1)
    plt.plot(iteration_arr, b_residual_norm_arr, "-o")
    plt.xlabel("iterations")
    plt.ylabel("norm of b residual")
    plt.yscale("log")
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(iteration_arr, f_arr, "-o")
    plt.xlabel("iterations")
    plt.ylabel("f(x)")
    plt.yscale("log")
    plt.grid(True)

    optimized_guess = current_x
    return optimized_guess, iterations


if __name__ == "__main__":
    optimized_guess, iterations_taken = SteepestDescent()
    print("Took", iterations_taken, "iterations")
    print("optimized_guess: ", optimized_guess)
    visualize_f(-10, 10, -10, 10)
    plt.show()
