"""
Goal of this code is to solve the following system of nonlinear equations for variables x1, x2, ..., xn:

x1 = f1(x1, x2, ..., xn)
x2 = f2(x1, x2, ..., xn)
            .
            .
            .
xn = fn(x1, x2, ..., xn)

The above can be more compactly represented by the nonlinear vector equation: x = f(x)

The example below tries to solve a system describing a series resistor (220 ohms) and a diode circuit powered by 5V.

x is a vector with decision variables:
 x1: voltage drop across the LED
 x2: current in the circuit

f(x) returns the vector:
    f1(x): voltage drop across the LED
    f2(x): current through the LED (same as the current through the circuit)

"""

import matplotlib.ticker as mticker
import numpy as np
from math import exp, log
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def f(x):
    result = np.zeros(np.shape(x))
    result[0] = 5 - 220 * x[1]
    result[1] = 1e-12 * (exp(x[0] / 25.85e-3) - 1)
    return result


def loss(x):
    """
    The problem to be solved is to find the vector x such that x = f(x).
    Equivalently, I want to minimize the "difference" between x and f(x).
    The L2 loss used below quantifies this "difference" in the vector sense
    """
    return np.linalg.norm(x - f(x))


def visualizeLoss(lossFunction, min_x1, max_x1, min_x2, max_x2):
    x1_arr = np.linspace(min_x1, max_x1, 50)
    x2_arr = np.linspace(min_x2, max_x2, 50)

    X, Y = np.meshgrid(x1_arr, x2_arr)
    Z = np.zeros(np.shape(X))
    for i in range(np.shape(Z)[0]):
        for j in range(np.shape(Z)[1]):
            Z[i, j] = lossFunction(np.array([[X[i, j], Y[i, j]]]).T)

    ax = plt.figure().gca(projection=Axes3D.name)
    Z = np.log10(Z)
    ax.plot_surface(X, Y, Z, cmap='viridis')
    ax.set_title('Loss')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('loss')

    # https://github.com/matplotlib/matplotlib/issues/209
    def log_tick_formatter(val, pos=None):
        return r"$10^{{{:.0f}}}$".format(val)

    ax.zaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))


def cos_sim(a, b):
    dot_product = np.dot(a.flatten(), b.flatten())  # x dot y
    norm_a = np.linalg.norm(a)  # ||x||
    norm_b = np.linalg.norm(b)  # ||b||
    return dot_product / (norm_a * norm_b)


def gradientDescent(lossFunction, initial_guess, max_iterations=100, loss_threshold=1e-5, step_size=1e-2):
    def gradient(x):
        h = 1e-12  # step size used to approximate the gradient of the loss function
        x1 = x.astype(np.float64)
        grad = np.zeros(np.shape(x))
        for i in range(np.shape(x)[0]):
            x2 = np.copy(x)
            x2[i] += h
            grad[i] = (lossFunction(x2) - lossFunction(x1)) / h
        return grad

    previous_x = initial_guess
    previous_grad = gradient(previous_x)
    previous_loss = lossFunction(previous_x)

    current_x = previous_x
    current_grad = previous_grad
    current_loss = previous_loss

    iteration = 0

    loss_arr = [current_loss]
    step_size_arr = [step_size]
    iterations = [iteration]

    loss_ratio = 1e99

    """early stopping conditions:
    0) completed the maximum number of iterations 
    1) the current loss becomes acceptable because it falls below the loss threshold 
    2) the loss barely changes. Mathematically, the ratio between the current to previous loss is very close to 1. Thus,
    the absolute value of the log of that ratio is close to 0) """
    while iteration < max_iterations and current_loss > loss_threshold and abs(log(loss_ratio)) > 1e-4:

        current_x -= current_grad * step_size
        current_grad = gradient(current_x)
        current_loss = lossFunction(current_x)

        gradient_progress = cos_sim(previous_grad, current_grad)
        if gradient_progress < -0.7:
            """if the previous and current gradients are in heavily opposite directions (cosine similarity between 
            the gradients being less than -0.9, that means the solution could be bouncing around a minimum. Thus, 
            I should decrease the step size and redo the iteration with a reduced step size.
            """
            step_size *= 0.1
            current_x = previous_x - current_grad * step_size
            current_grad = gradient(current_x)
            current_loss = lossFunction(current_x)

        loss_ratio = current_loss / previous_loss

        iteration += 1

        loss_arr.append(current_loss)
        step_size_arr.append(step_size)
        iterations.append(iteration)

        previous_x = current_x
        previous_grad = current_grad
        previous_loss = current_loss

    plt.subplot(2, 1, 1)
    plt.plot(iterations, loss_arr, "r-")
    plt.xlabel("iterations")
    plt.ylabel("loss")
    plt.yscale("log")
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(iterations, step_size_arr, "r-")
    plt.xlabel("iterations")
    plt.ylabel("step size")
    plt.yscale("log")
    plt.grid(True)

    return iteration, current_x


if __name__ == "__main__":
    iters, x = gradientDescent(loss, np.array([[0.6, 0.02]]).astype(np.float64).T, 10000, 1e-10, 1e-2)
    print("Took {} iterations. new x = {}".format(iters, x))
    visualizeLoss(loss, -0.9, 0.9, -0.1, 0.1)
    plt.show()
