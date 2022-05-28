"""
This code fits a SISO (Single variable Input and Single variable Output) model
The model is a * (1 - exp(-b * x)) where x, y, a, b are all scalars
a and b are the parameters to be fit
"""

import numpy as np
from math import exp
from matplotlib import pyplot as plt

from GaussNewtonUtils import GaussNewton

# generating dataset
x_data = [10, 40, 80, 140, 200, 300]
y_data = [3.1, 11.9, 21.0, 29.9, 37.3, 42.7]

parameters_initial_guess = np.array([[40, 5e-3]]).T


def model(params, x):
    a = params[0, 0]
    b = params[1, 0]
    return a * (1 - exp(-b * x))


def model_gradient(params, x):
    gradient = []

    a = params[0, 0]
    b = params[1, 0]
    gradient.append(1 - exp(-b * x))
    gradient.append(a * x * exp(-b * x))
    return np.array(gradient)


if __name__ == "__main__":
    GN = GaussNewton(x_data, y_data, model, model_gradient)

    GN.plot_prediction(parameters_initial_guess, title="initial guess")

    optimized_params, iterations, loss_arr = GN.Optimize(parameters_initial_guess, 100, 1e-9)
    print("optimized_params: ", optimized_params)
    plt.figure()
    plt.plot(iterations, loss_arr, "yo-")
    plt.yscale('log')
    plt.xlabel("iterations")
    plt.ylabel("loss")
    plt.grid(True)

    GN.plot_prediction(optimized_params, title="optimized parameters")

    plt.show()
