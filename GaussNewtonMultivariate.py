# GOAL: fit a nonlinear model parameteried by (l) params that predicts y (R^m) for a given x (R^n)

import numpy as np
from matplotlib import pyplot as plt

np.random.seed(0)

# ====================================================================
# start code custom to each optimization problem

# generating mock dataset
A_true = np.array([[1.4, 2],
                   [1, 1.4],
                   [0.5, 5]])

x_data = [np.random.rand(2, 1) for i in range(5000)]
y_data = [A_true @ x + np.random.multivariate_normal(np.zeros(3), 0.05 * np.eye(3), 1).T for x in x_data]

A_guess = np.array([[0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0]])

parameters_initial_guess = A_guess.flatten().reshape(-1, 1)

# end code custom to each optimization problem
# ====================================================================


NUM_PARAMS = parameters_initial_guess.size

assert (len(x_data) == len(y_data))
NUM_DATA_POINTS = len(x_data)

X_DIM = x_data[0].size
Y_DIM = y_data[0].size


# ====================================================================
# start code custom to each optimization problem

def model_gradient(params, x_i):
    """
    :param params: vector of parameters that parameterize the model
    :param x_i: a single multidimensional datapoint
    :return: (mxl) gradients of each output entry wrt params
    """
    gradient = np.array([[x_i[0, 0], x_i[1, 0], 0, 0, 0, 0],
                         [0, 0, x_i[0, 0], x_i[1, 0], 0, 0],
                         [0, 0, 0, 0, x_i[0, 0], x_i[1, 0]]])

    assert (gradient.shape == (Y_DIM, NUM_PARAMS))
    return gradient


def model(params, x):
    A = params.reshape(A_guess.shape)
    return A @ x


# end code custom to each optimization problem
# ====================================================================

def model_predict_dataset(params, x):
    """
    :param params: vector of parameters that parameterize the model
    :param x: list of x inputs, each of which has dimension n
    :return: list of y outputs, each of which has dimension m
    """
    return [model(params, x_i) for x_i in x]


def get_residuals(params, y):
    # y_data is a list of numpy arrays, need it to be a complete numpy array though
    y_np_array = np.concatenate(y, axis=0)

    predictions = model_predict_dataset(params, x_data)
    predictions_np_array = np.concatenate(predictions, axis=0)

    residuals = y_np_array - predictions_np_array
    return residuals.flatten().reshape(-1, 1)


def calculate_loss(residuals):
    return (residuals.T @ residuals)[0, 0]


# def plot_prediction(params):
#     y_predicted = model_predict_dataset(params, x_data)
#     plt.plot(x_data[0], y_predicted[0], "r*-")  # prediction
#     plt.scatter(x_data, y_data, color="g", marker="*")  # ground truth
#     plt.show()


def get_jacobian(params, x):
    """
    :param params: vector of parameters that parameterize the model
    :param x: list of x inputs, each of which has dimension n
    :return: Jacobian of the model for each input
    """
    J = np.zeros((NUM_DATA_POINTS * Y_DIM, NUM_PARAMS))
    for i in range(NUM_DATA_POINTS):
        J[Y_DIM * i:Y_DIM * i + Y_DIM] = model_gradient(params, x[i])
    return J


def GaussNewton(init_params, x, y, max_iters, loss_thresh):
    params = init_params

    iteration = 0

    prev_loss = 1e99
    curr_loss = calculate_loss(get_residuals(params, y))

    iterations = [iteration]
    loss_arr = [curr_loss]

    r = get_residuals(params, y)
    while iteration < max_iters and abs(curr_loss - prev_loss) > loss_thresh:
        J = get_jacobian(params, x)
        delta_params = np.linalg.inv(J.T @ J) @ J.T @ r

        params += delta_params
        iteration += 1

        prev_loss = curr_loss
        r = get_residuals(params, y)
        curr_loss = calculate_loss(r)

        iterations.append(iteration)
        loss_arr.append(curr_loss)

    # plot_prediction(params)
    plt.plot(iterations, loss_arr, "yo-")
    plt.yscale('log')
    plt.xlabel("iterations")
    plt.ylabel("loss")
    plt.grid(True)
    plt.show()
    return iteration, params


iterations_took, fitted_params = GaussNewton(parameters_initial_guess, x_data, y_data, 100, 1e-9)
print(fitted_params.reshape(A_guess.shape))
