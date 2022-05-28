"""
This code fits a MIMO (Multiple variable Input and Multiple variable Output) model.
The model is y=Ax for 3x1 y, 3x2 A (to be fit), 2x1 x
The entries of matrix 'A' are the parameters to be fit

Ideally, the matrix recovered by the optimization algorithm (A_guess) is
close to the matrix used to generate the dataset (A_true) despite the noise
that was added the output datapoints (y_data)
"""

import numpy as np
from matplotlib import pyplot as plt

from GaussNewtonUtils import GaussNewton

np.random.seed(0)

# generating dataset
A_true = np.array([[1.4, 2],
                   [1, 1.4],
                   [0.5, 5]])

x_data = [np.random.rand(2, 1) for i in range(5000)]
y_data = [A_true @ x + np.random.multivariate_normal(np.zeros(3), np.eye(3), 1).T for x in x_data]

A_guess = np.array([[-231.0, 40.0],
                    [50.0, -10.0],
                    [10.0, 10.0]])

parameters_initial_guess = A_guess.flatten().reshape(-1, 1)


def model(params, x):
    A = params.reshape(A_guess.shape)
    return A @ x


def model_gradient(params, x):
    return np.array([[x[0, 0], x[1, 0], 0, 0, 0, 0],
                     [0, 0, x[0, 0], x[1, 0], 0, 0],
                     [0, 0, 0, 0, x[0, 0], x[1, 0]]])


if __name__ == "__main__":
    GN = GaussNewton(x_data, y_data, model, model_gradient)

    optimized_params, iterations, loss_arr = GN.Optimize(parameters_initial_guess, 100, 1e-9)
    print("optimized_params: ", optimized_params.reshape(A_guess.shape))
    plt.plot(iterations, loss_arr, "yo-")
    plt.yscale('log')
    plt.xlabel("iterations")
    plt.ylabel("loss")
    plt.grid(True)
    plt.show()
