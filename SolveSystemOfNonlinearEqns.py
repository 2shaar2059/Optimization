import numpy as np
from math import exp, sin
from scipy.spatial.distance import cosine
from matplotlib import pyplot as plt

def f(x):

	result = np.zeros(np.shape(x))

	result[0] = 5 - 10 * x[1]
	result[1] = 1e-12 * (exp(x[0] / 25.85e-3) - 1)
	#result[0] = x[1]**2
	#result[1] = 2  #sin(x[0])+1
	return result


def loss(x):
	return np.linalg.norm(x - f(x))

def visualizeLoss(loss, minx, maxx, miny, maxy):
	x = np.linspace(minx, maxx, 50)
	y = np.linspace(miny, maxy, 50)

	X, Y = np.meshgrid(x, y)
	Z = np.zeros(np.shape(X))
	for i in range(np.shape(Z)[0]):
		for j in range(np.shape(Z)[1]):
			Z[i,j] = loss(np.array([[X[i,j], Y[i,j]]]).T)


	fig = plt.figure()
	ax = plt.axes(projection='3d')
	ax.plot_surface(X, Y, Z, cmap='viridis')
	ax.set_title('Loss')
	ax.set_xlabel('X')
	ax.set_ylabel('Y')
	ax.set_zlabel('Z')

	plt.show()


def gradientDescent(loss, xinit, maxIters, threshhold, alpha):
	def getGradient(loss, x):
		h = 1e-12  #step size used to approximate the gradient of the loss function
		x1 = x.astype(np.float64)
		del_f = np.zeros(np.shape(x))
		for i in range(np.shape(x)[0]):
			x2 = np.copy(x)
			x2[i] += h
			del_f[i] = (loss(x2) - loss(x1)) / h
		return (del_f)

	cos_sim_arr = []
	loss_arr = []
	alpha_arr = []
	cnt = []

	x = xinit

	del_f_prev = np.zeros(np.shape(x))
	x_prev = np.zeros(np.shape(x))

	counter = 0
	while counter < maxIters and loss(x) > threshhold:
		"""
		TODO: if the current gradient points opposite to the previosu gradient, redo the previous iteration with a reduced alpha. This way if you overshoot and the gradient points opposite to how it used to, you dont have to go back with a small alpha

		"""


		del_f = getGradient(loss, x)

		"""#if the previous and current gradients are in 
		opposite directions, that means the solution is 
		bouncing around the minimum so we need to lower 
		the alpha step size
		"""
		"""

		if((1 -cosine(del_f_prev, del_f)) < -0.9):
			alpha*=0.1
			x = (x_prev - getGradient(loss, x_prev) * alpha)
		else:
			x -= del_f * alpha
		"""
		if((1 -cosine(del_f_prev, del_f)) < -0.9):
			alpha*=0.1
		x -= del_f * alpha


		cos_sim_arr.append(1 -cosine(del_f_prev, del_f))
		loss_arr.append(loss(x))
		alpha_arr.append(alpha)
		cnt.append(counter)

		print("x:{}\n loss:{}\n gradient:{}\n".format(x, loss(x), del_f))

		counter += 1
		del_f_prev = del_f
		x_prev = x


	plt.subplot(2,2,1)
	plt.plot(cnt, cos_sim_arr, "r-")
	plt.xlabel("iterations")
	plt.ylabel("cos similarity between previous and current gradient")

	plt.subplot(2,2,2)
	plt.plot(cnt, loss_arr, "r-")
	plt.xlabel("iterations")
	plt.ylabel("loss")
	plt.yscale("log")

	plt.subplot(2,2,3)
	plt.plot(cnt, alpha_arr, "r-")
	plt.xlabel("iterations")
	plt.ylabel("alpha")
	plt.yscale("log")

	plt.show()


	return counter, x


iters, x = gradientDescent(loss, np.array([[0.70, 0.4307]]).astype(np.float64).T, 200, 1e-10, 6e-1)
#iters, x = gradientDescent(loss, np.array([[1,2]]).astype(np.float64).T, 10000, 1e-10, 1e-1)
print("Took {} iterations. new x = {}".format(iters, x))
#visualizeLoss (loss, -0.7, 0.7, -1, 1)