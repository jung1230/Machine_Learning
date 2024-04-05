import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cvx
from numpy.matlib import repmat

print("NumPy version:", np.__version__)
print("CVXPY version:", cvx.__version__)
'''
h = 1
c0 = np.loadtxt("data/homework4_class0.txt")
c1 = np.loadtxt("data/homework4_class1.txt")

x = np. vstack (( c0 , c1 ))
c0 = np.array([np.zeros(c0.shape[0])]).T
c1 = np.array([np.ones(c1.shape[0])]).T
y = np.vstack([c0,c1])


# your code
K = np.zeros ((100 ,100))
h=1
for i in range(100):
  for j in range(100):
    K[i,j] = np.exp(-np.linalg.norm((x[i,:]-x[j,:]))**2/h)
print(K[47:52 ,47:52])



# your code 
N = 100
lmda = 0.0001
alpha = cvx.Variable ((100 ,1))
e0_eka = cvx.hstack([np.zeros((N,1)), K@alpha])
loss = - y.T@K@alpha + cvx.sum(cvx.log_sum_exp(e0_eka , axis=1))
reg = cvx.quad_form(alpha,K) #aˆTKa
prob = cvx.Problem(cvx.Minimize(loss/N + lmda*reg))
prob.solve ()
alpha = alpha.value
print(f"alpha = \n{alpha[0:2]}")





'''