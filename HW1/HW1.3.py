
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.special import eval_legendre
from scipy.optimize import linprog

# ------------------------- Exercise 3. -------------------------
# --------------- (a) ---------------
# Set random seed for reproducibility
np.random.seed(22) 

x = np.linspace(-1,1,50) # 50 points in the interval [-1,1], 50 rows in total

# Gaussian error term
epsilon = np.random.normal(0, 0.2**2, size=len(x))  

y_true = -0.001 + 0.01 *  eval_legendre(1,x) + 0.55 *  eval_legendre(2,x) + 1.5 *  eval_legendre(3,x) + 1.2 *  eval_legendre(4,x) 
y = y_true + epsilon

plt.scatter(x, y, label='Noisy Data')
plt.plot(x, y_true, label='True Model', color='red', linestyle='dashed')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()

# --------------- (b) ---------------
# B hat = (X transpose * X)inverse * X transpose * y
# X = data
# B = theta
# y = B Transpose * X

# Theta = np.linalg.lstsq(A_legendre, y, rcond=None)[0] # do not compute (A^TA)^-1 Yourself!

# --------------- (c) ---------------
A_legendre =  np.column_stack((eval_legendre(0,x), eval_legendre(1,x), eval_legendre(2,x), eval_legendre(3,x), eval_legendre(4,x)))

# np.linalg.lstsq Return the !!!least-squares solution!!! to a linear matrix equatio
Theta = np.linalg.lstsq(A_legendre, y, rcond=None)[0]

y_predicted = np.dot(A_legendre, Theta)
plt.scatter(x, y, label='Noisy Data')
plt.plot(x, y_true, label='True Model', color='red', linestyle='dashed')
plt.plot(x, y_predicted, label='Predicted Model', color='green')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()

# --------------- (d) ---------------
idx = [10,16,23,37,45] # these are the locations of the outliers
y[idx] = 5

# np.linalg.lstsq Return the least-squares solution to a linear matrix equatio
Theta = np.linalg.lstsq(A_legendre, y, rcond=None)[0]

y_predicted = np.dot(A_legendre, Theta)
plt.scatter(x, y, label='Noisy Data')
plt.plot(x, y_true, label='True Model', color='red', linestyle='dashed')
plt.plot(x, y_predicted, label='Predicted Model', color='green')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
# --------------- (e) ---------------
# c = [[0d],[1N]]
# x = [[theta],[mu]]
# A = [[X, -I],[-X, -I]]
# b = [[y],[-y]]
# !!! put those matrix in "np.array()" !!!

# --------------- (f) ---------------
# hstack: Stack arrays in sequence horizontally (column wise, like expanding the matrix).
c_t = np.hstack((np.zeros(5), np.ones(50)))

# https://stackoverflow.com/questions/16473042/numpy-vstack-vs-column-stack
X = np.column_stack((eval_legendre(0,x), eval_legendre(1,x), eval_legendre(2,x), eval_legendre(3,x), eval_legendre(4,x)))

# vstack: Stack arrays in sequence vertically (row wise).
# eye: Return a 2-D array with ones on the diagonal and zeros elsewhere. (input = row or column)
A = np.vstack((np.hstack((X, -np.eye(50))), np.hstack((-X, -np.eye(50)))))
b = np.hstack((y, -y))

res = linprog(c_t, A, b, bounds=(None, None), method='highs')
theta = res.x[:5]  # Extract theta from the solution

# Calculate the predicted values using the obtained theta
y_predicted = theta[0] * eval_legendre(0,x) + theta[1] * eval_legendre(1,x) + theta[2] * eval_legendre(2,x) + theta[3] * eval_legendre(3,x) + theta[4] * eval_legendre(4,x)


# Scatter plot the data and overlay with the predicted curve
plt.scatter(x, y, label='Noisy Data')
plt.plot(x, y_true, label='True Model', color='red', linestyle='dashed')
plt.plot(x, y_predicted, label='Predicted Model', color='green')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()