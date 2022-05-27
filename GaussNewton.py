#GOAL: fit a nonlinear model that predicts y for a given x

"""Helpful links: 
https://en.wikipedia.org/wiki/Gauss%E2%80%93Newton_algorithm
https://www.youtube.com/watch?v=Sc3dgcENDAI
"""

from matplotlib import pyplot as plt
from math import exp
import numpy as np

#====================================================================
# start of the code that is custom to each optimization problem

x_data = np.array([[10,40,80,140,200,300]])
y_data = np.array([[3.1, 11.9,21.0,29.9,37.3,42.7]])

parameters_initial_guess = np.array([[40, 5e-3]]).T
parameters = parameters_initial_guess

NUM_PARAMS = np.shape(parameters_initial_guess)[0]

assert(np.shape(x_data)[1]==np.shape(y_data)[1])
NUM_DATA_POINTS = np.shape(x_data)[1]

def model_gradient(params, x):
	gradient = np.zeros((NUM_PARAMS))

	a = params[0,0]
	b = params[1,0]
	gradient[0]=1-exp(-b*x)
	gradient[1]=a*x*exp(-b*x)
	return gradient

def model(params, x):
	a = params[0, 0]
	b = params[1, 0]
	return a*(1-exp(-b*x))

# end of the code that is custom to each optimization problem
#====================================================================

def model_predict_dataset(params, x):
	predictions = np.zeros((1, NUM_DATA_POINTS))
	for i in range(0, NUM_DATA_POINTS):
		predictions[0, i] = model(params, x[0, i])
	return predictions

def get_residuals(params):
	residuals = y_data-model_predict_dataset(params, x_data)
	return residuals.T

def calculate_loss(residuals):
	return (residuals.T@residuals)[0,0]

def plot_prediciton(params):
	y_predicted = model_predict_dataset(params, x_data)
	plt.plot(x_data[0], y_predicted[0], "r*-") # prediction
	plt.scatter(x_data,y_data, color="g", marker="*") # ground truth
	plt.show()

def get_jacobian(params, x):
	jacobian = np.zeros((NUM_DATA_POINTS, NUM_PARAMS))
	for i in range(NUM_DATA_POINTS):
		jacobian[i] = model_gradient(params, x[0, i])
	return jacobian

def GaussNewton(init_params, x, y, maxIters, loss_thresh):
	params=init_params

	iteration = 0
	
	prev_loss = 1e99
	curr_loss = calculate_loss(get_residuals(params))

	iterations = [iteration]
	loss_arr = [curr_loss]

	while iteration < maxIters and abs(curr_loss-prev_loss) > loss_thresh:

		J = get_jacobian(params, x)
		r = get_residuals(params)		
		delta_params = np.linalg.inv(J.T@J)@J.T@r

		params+=delta_params
		iteration += 1

		prev_loss = curr_loss
		curr_loss = calculate_loss(get_residuals(params))

		iterations.append(iteration)
		loss_arr.append(curr_loss)

	plot_prediciton(params)
	plt.plot(iterations, loss_arr, "yo-")
	plt.yscale('log')
	plt.xlabel("iterations")
	plt.ylabel("loss")
	plt.grid(True)
	plt.show()
	return iteration

GaussNewton(parameters_initial_guess, x_data, y_data, 100, 1e-9)