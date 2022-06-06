"""This code solves Mx=y for x by iteratively minimizing the associated quadratic form via steepest descent. """

import numpy as np
from matplotlib import pyplot as plt

# ====================================================================
# start of the code that is custom to each optimization problem

M = np.array([[1.0, 2.0],
              [1.0, 4.0],
              [5.0, 6.0]])

y = np.array([[7.0],
              [8.0],
              [9.0]])

# initial guess
x0 = np.array([[0.0],
               [0.0]])

"""once the norm of the residual falls below this tolerance, stop optimizing"""
RESIDUAL_TOLERANCE = 1e-10

MAX_ITERATIONS = 10

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
    return 0.5 * (y_residual.T @ y_residual)[0][0]


def visualize_f(x_arr, f_arr):
    x1_arr = np.linspace(-10, 10, 80)
    x2_arr = np.linspace(-10, 10, 80)

    X, Y = np.meshgrid(x1_arr, x2_arr)
    Z = np.zeros(np.shape(X))
    for i in range(np.shape(Z)[0]):
        for j in range(np.shape(Z)[1]):
            Z[i, j] = f(np.array([[X[i, j], Y[i, j]]]).T)
    #
    # ax = plt.figure(2).gca(projection=Axes3D.name)
    # ax.plot_surface(X, Y, Z, cmap='viridis')
    # ax.set_title('f(x)')
    # ax.set_xlabel('x1')
    # ax.set_ylabel('x2')
    # ax.set_zlabel('f')
    # ax.plot(x1_arr, x2_arr, f_arr, "-o")
    plt.figure(3)
    plt.contour(X, Y, Z, 70)
    plt.grid(True)
    plt.colorbar()
    plt.gca().set_aspect('equal')
    x1_arr = []
    x2_arr = []

    for x in x_arr:
        x1_arr.append(x[0][0])
        x2_arr.append(x[1][0])
    plt.plot(x1_arr, x2_arr, "-o")


def SteepestDescent():
    current_x = x0
    prev_residual = None
    residual = b - A @ current_x
    residual_norm = np.linalg.norm(residual)
    iterations = 0

    b_residual_norm_arr = [residual_norm]
    x_arr = [np.copy(current_x)]
    f_arr = [f(current_x)]
    iteration_arr = [0]

    while iterations < MAX_ITERATIONS and residual_norm > RESIDUAL_TOLERANCE:
        gradient = (-residual).T
        step_size = -gradient @ gradient.T / (gradient @ A @ gradient.T)

        current_x += step_size * gradient.T
        prev_residual = residual
        residual = b - A @ current_x
        residual_norm = np.linalg.norm(residual, 2)
        iterations += 1

        print("gradient:", gradient)
        print("step_size:", step_size)
        print("new x:", current_x)
        print("new y residual (r_k):", y - M @ current_x)
        print("new b residual (r_k):", residual)
        print("r_k-1 dot r_k:", prev_residual.T @ residual)
        print("Iterations Completed:", iterations)
        print()

        b_residual_norm_arr.append(residual_norm)
        x_arr.append(np.copy(current_x))
        f_arr.append(f(current_x))
        iteration_arr.append(iterations)

    plt.figure(1)
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
    return optimized_guess, x_arr, f_arr, iterations


if __name__ == "__main__":
    optimized_guess, x_arr, f_arr, iterations_taken = SteepestDescent()
    print("Took", iterations_taken, "iterations")
    print("optimized_guess: ", optimized_guess)
    visualize_f(x_arr, f_arr)
    plt.show()
