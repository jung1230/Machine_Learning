import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cvx
from numpy.matlib import repmat

# ---------------- Exercise 2 ----------------
# --------------------- (b)---------------------

xclass0 = np.matrix(np.loadtxt('homework4_class0.txt'))
xclass1 = np.matrix(np.loadtxt('homework4_class1.txt'))
x = np.vstack((xclass0, xclass1))

[rowx0,colx0] = np.shape(xclass0)
[rowx1,colx1] = np.shape(xclass1)
yclass0 = np.zeros((rowx0,1))
yclass1 = np.ones((rowx1,1))
y = np.vstack((yclass0, yclass1))

[rowx,colx] = np.shape(x) #(100, 2)
x = np.hstack((x, np.ones((rowx, 1))))
N = rowx
theta = cvx.Variable((3, 1))
lambdaa = 0.0001

# cvx.sum performs more than just a simple sum, works like sigma.
# For cvx.log_sum_exp, see note in week7 P16
# The axis=1 argument indicates that the operation should be applied along the rows of the matrix. This means that the log-sum-exp is computed for each row individually, resulting in a vector where each element corresponds to the log-sum-exp of the corresponding row.
LossTerm  = (cvx.sum(cvx.multiply(y, x @ theta)) - cvx.sum(cvx.log_sum_exp(cvx.hstack([np.zeros((N,1)), x @ theta]), axis=1 ))) / (-N)

RegularizationTerm = cvx.sum_squares(theta) * lambdaa
Problem = cvx.Problem(cvx.Minimize(LossTerm + RegularizationTerm))
Problem.solve(solver=cvx.ECOS)
theta_hat = theta.value
print(theta_hat)



# --------------------- (c)---------------------

plt.scatter(xclass0[:,0].tolist(),xclass0[:,1].tolist(), c='b')
plt.scatter(xclass1[:,0].tolist(),xclass1[:,1].tolist(), c='g')

# decision boundary: 0 = θ0*x0 + θ1*x1 + θ2
x0 = np.linspace(-5,10,10)
x1 = (-theta_hat[0]*x0 - theta_hat[2]) / theta_hat[1]

plt.plot(x0,x1)
plt.show()



# --------------------- (d)---------------------
# functions of bayesian decision can be found in hw3.pdf and hw3.py Exercise2a

# testing sites
TestRange = np.linspace(-5, 10, N)

decision_boundary = np.zeros((N, N))

u0 = np.zeros(colx0)
for i in range(colx0):
    u0[i] = np.mean(xclass0[i,:])
u1 = np.zeros(colx1)
for i in range(colx1):
    u1[i] = np.mean(xclass1[i,:])
sigma0 = np.cov(xclass0.T)
sigma1 = np.cov(xclass1.T) # .T?????????
abs_Sigma1 = np.linalg.det(sigma1)
abs_Sigma0 = np.linalg.det(sigma0)
inv_Sigma1 = np.linalg.inv(sigma1)
inv_Sigma0 = np.linalg.inv(sigma0)
pi1 = rowx1 / (rowx0 + rowx1)
pi0 = rowx0 / (rowx0 + rowx1)

for i in range(N):
    for j in range(N):
        block = np.matrix([TestRange[i], TestRange[j]]).T
        c1 = (-1/2) * (block - u1).T @ np.linalg.inv(sigma1) @ (block - u1) + np.log(pi1) - (1/2) * np.log(np.linalg.det(sigma1)) # equation from 2a
        c0 = (-1/2) * (block - u0).T @ np.linalg.inv(sigma0) @ (block - u0) + np.log(pi0) - (1/2) * np.log(np.linalg.det(sigma0))
        if np.sum(c1) < np.sum(c0): # compare the total cost
            decision_boundary[i, j] = 1
plt.scatter(xclass0[:,0].tolist(),xclass0[:,1].tolist(), c='b')
plt.scatter(xclass1[:,0].tolist(),xclass1[:,1].tolist(), c='g')
plt.contour(TestRange,TestRange,decision_boundary)
plt.show()



# ---------------- Exercise 3 ----------------
# --------------------- (a)---------------------

h = 1
K = np.zeros((N,N))
x = x[:,0:2] # remove the additional column of 1 added in line 22( x = np.hstack((x, np.ones((rowx, 1)))) )

for i in range(N):
    for j in range(N):
        K[i, j] = np.exp(-np.linalg.norm(x[i] - x[j]) ** 2 / h)
print(K[47:52, 47:52])



# --------------------- (c)---------------------

alpha = cvx.Variable((N, 1))
loss = - y.T @ K @ alpha + cvx.sum(cvx.log_sum_exp( cvx.hstack([np.zeros((N, 1)), K @ alpha]), axis=1 ))
RegularizationTerm = cvx.quad_form(alpha, K)
problem = cvx.Problem(cvx.Minimize(loss/N + lambdaa * RegularizationTerm))
problem.solve()
ALPHA = alpha.value
print(ALPHA[:2])



# --------------------- (d)---------------------

# testing sites
TestRange = np.linspace(-5, 10, N)

decision_boundary = np.zeros((N, N))
x = np.hstack((x, np.ones((rowx, 1))))

for i in range(N):
    for j in range(N):
        data = repmat(np.array([TestRange[j], TestRange[i], 1]).reshape((1, 3)), N, 1)
        phi  = np.exp( -np.sum( (np.array(x)-data)**2, axis=1)/h )
        decision_boundary[i,j] = np.dot(phi.T, ALPHA)

plt.scatter(xclass0[:,0].tolist(), xclass0[:,1].tolist(), c='b')
plt.scatter(xclass1[:,0].tolist(), xclass1[:,1].tolist(), c='g')
plt.contour(TestRange, TestRange, decision_boundary) 
# plt.contour(TestRange, TestRange, decision_boundary > 0.5) # set 0.5 to make it only one boundary
plt.show()