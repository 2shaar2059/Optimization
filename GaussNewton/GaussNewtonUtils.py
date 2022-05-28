# GOAL: fit a nonlinear model that predicts y for a given x

"""Helpful links: 
https://en.wikipedia.org/wiki/Gauss%E2%80%93Newton_algorithm
https://www.youtube.com/watch?v=Sc3dgcENDAI
"""

import numpy as np
from math import log
from matplotlib import pyplot as plt


def calculate_l2_loss(residuals):
    return (residuals.T @ residuals)[0, 0] / len(residuals)


class GaussNewton:
    def __init__(self, x_data, y_data, model, model_gradient):
        """
        :param x_data: list of input datapoints, each of which has dimension (nx1)
        :param y_data: list of output datapoints, each of which has dimension (mx1)
        :param model: function that takes in 1) a vector of parameters
        (lx1) that parameterize the model and 2) a single input datapoint. Should return a (mx1) output datapoint (
        y-value)
        :param model_gradient: function that takes in 1) a vector of parameters (lx1) that parameterize the
        model and 2) a single input datapoint. Should return m, (lx1) gradients of each of the m dimensions of y wrt
        params. This return value should be formatted as an (mxl) numpy array
        """
        self.NUM_PARAMS = None
        self.parameters_initial_guess = None
        self.model = model
        self.model_gradient = model_gradient

        assert type(x_data) == list and type(y_data) == list
        assert len(x_data) == len(y_data)

        self.x_data = x_data
        self.y_data = y_data
        self.X_DIM = x_data[0].size if isinstance(x_data[0], np.ndarray) else 1
        self.Y_DIM = y_data[0].size if isinstance(y_data[0], np.ndarray) else 1
        self.NUM_DATA_POINTS = len(x_data)

    def model_predict_on_dataset(self, params):
        return [self.model(params, x) for x in self.x_data]

    def calculate_residuals(self, params):
        # y_data is a list of numpy arrays but need it to be a numpy array of numpy arrays
        y_np_array = np.array(self.y_data)

        predictions = self.model_predict_on_dataset(params)
        predictions_np_array = np.array(predictions)

        residuals = y_np_array - predictions_np_array
        return residuals.reshape(-1, 1)

    def calculate_jacobian(self, params):
        """
        :param params: vector of parameters that parameterize the model
        :return: Jacobian of the model for each input
        """
        J = np.zeros((self.NUM_DATA_POINTS * self.Y_DIM, self.NUM_PARAMS))
        for i in range(self.NUM_DATA_POINTS):
            gradient = self.model_gradient(params, self.x_data[i])
            if self.Y_DIM == 1:
                assert gradient.shape[0] == self.NUM_PARAMS
            else:
                assert gradient.shape == (self.Y_DIM, self.NUM_PARAMS)
            J[self.Y_DIM * i:self.Y_DIM * i + self.Y_DIM] = gradient
        return J

    def Optimize(self, parameters_initial_guess, max_iterations, loss_threshold):
        self.parameters_initial_guess = parameters_initial_guess
        self.NUM_PARAMS = len(parameters_initial_guess)
        params = self.parameters_initial_guess
        iteration = 0
        previous_loss = 1e99
        residuals = self.calculate_residuals(params)
        current_loss = calculate_l2_loss(residuals)

        iterations = [iteration]
        loss_arr = [current_loss]

        loss_ratio = 1e99

        """early stopping conditions:
        0) completed the maximum number of iterations 
        1) the current loss becomes acceptable because it falls below the loss threshold 
        2) the loss barely changes. Mathematically, the ratio between the current to previous loss is very close to 1. Thus,
        the absolute value of the log of that ratio is close to 0) """
        while iteration < max_iterations and current_loss > loss_threshold and abs(log(loss_ratio)) > 1e-7:
            loss_ratio = current_loss / previous_loss
            J = self.calculate_jacobian(params)
            delta_params = np.linalg.inv(J.T @ J) @ J.T @ residuals

            params += delta_params
            previous_loss = current_loss
            residuals = self.calculate_residuals(params)
            current_loss = calculate_l2_loss(residuals)
            iteration += 1

            iterations.append(iteration)
            loss_arr.append(current_loss)

        return params, iterations, loss_arr

    def plot_prediction(self, params, title):
        if (self.X_DIM == 1 and self.Y_DIM == 1):
            y_predicted = self.model_predict_on_dataset(params)
            plt.figure()
            plt.plot(self.x_data, y_predicted, "r*-")  # prediction
            plt.scatter(self.x_data, self.y_data, color="g", marker="o")  # ground truth
            plt.grid(True)
            plt.xlabel("x")
            plt.ylabel("y")
            plt.title(title)
        else:
            print("Can't plot multivariate data")
