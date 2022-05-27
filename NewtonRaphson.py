from matplotlib import pyplot as plt
from math import sin, cos, exp, sqrt

xmin = -10
xmax = 10

def f(x):
	return 1 + (x%3 -1) * 10*x/(x*x+1)  + sin(x)*sqrt(abs(x)) - cos(4*x)*sin(2*x)

def graph(space):
	x = []
	y = []
	currX = xmin
	while currX < xmax:
		x.append(currX)
		y.append(f(currX))
		currX += space
	plt.plot(x,y,'b-')
	plt.grid()

def NewtonRaphson(xinit, maxIters, threshhold):
	slopeStep = 0.00000001
	currX = xinit
	currError = 999999
	prevError = 0
	counter = 0
	while counter < maxIters and abs(f(currX)) > threshhold:
		slope = (f(currX + slopeStep)-f(currX))/(slopeStep)
		if not slope == 0:
			currX += -f(currX)/slope
		currError = f(currX)
		counter += 1
		prevError = currError
	return counter

graph(0.00001)

step = 0.01
startX = []
endX = []
currX = xmin
while currX < xmax:
	startX.append(currX)
	endX.append(NewtonRaphson(currX, 100, 1e-9))
	currX+=step
plt.plot(startX, endX, "g*")
plt.show()