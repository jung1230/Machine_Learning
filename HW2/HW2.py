import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
import csv

# ---------------------------------------- Exercise 1: Loading Data via Python ----------------------------------------
print("Exercise 1")
# Reading csv file for female data
with open("female_train_data.csv", "r") as csv_file:
    reader = csv.reader(csv_file, delimiter=',')
    # normalize the number in female_stature_mm by dividing them with 1000
    # normalize that of female_bmi by dividing them with 10.

    counter = 0
    # Initialize modified_rows_female as an empty array
    modified_rows_female = np.array([])
    for row in reader:
        # skip the first row
        if counter == 0:
            modified_rows_female = np.array(row)
            counter += 1
            continue
        row[2] = float(row[2]) / 1000
        row[1] = float(row[1]) / 10
        modified_rows_female = np.row_stack((modified_rows_female, np.array(row, dtype=float)))
    
    # print the first 10 entries of female BMI
    # print the first 10 entries of female stature
    # print(modified_rows_female[0:11, :])

csv_file.close()

# Reading csv file for male data
with open("male_train_data.csv", "r") as csv_file:
    reader = csv.reader(csv_file, delimiter=',')
    # normalize the number in male_stature_mm by dividing them with 1000
    # normalize that of male_bmi by dividing them with 10.
    
    counter = 0
    # Initialize modified_rows_male as an empty array
    modified_rows_male = np.array([])
    for row in reader:
        # skip the first row
        if counter == 0:
            modified_rows_male = np.array(row)
            counter += 1
            continue
        row[2] = float(row[2]) / 1000
        row[1] = float(row[1]) / 10
        modified_rows_male = np.row_stack((modified_rows_male, np.array(row, dtype=float)))
    
    # print the first 10 entries of male BMI
    # print the first 10 entries of male stature
    # print(modified_rows_male[0:11, :])

csv_file.close()






print("\nExercise 2")
# ---------------------------------------- Exercise 2: Build a Linear Classifier via Optimization ----------------------------------------
# ---------------------------------------- (a) ----------------------------------------
# theta hat = (X^T * X)^(-1) * X^T * y. This solution is for unique global minimum in terms of the rank of X(week2 page9)

# Techniques when X^T * X is Not Invertible:
# 1. regularization: Ridge Regression(L2), LASSO(L1). from sklearn.linear_model import Lasso/Ridge (week3 page5, XTX + lambda*I is stable)
# 2. Pseudoinverse: numpy.linalg.pinv(). theta hat = (X^T * X)^(pseudoinvers) * X^T * y.


# ---------------------------------------- (b: analytic expression) ----------------------------------------
# Get A
# Round to decimals 3
data_m = np.around(modified_rows_male[1:,1:].astype(float), decimals = 3) 
data_f = np.around(modified_rows_female[1:,1:].astype(float), decimals = 3)
X = np.row_stack((data_m,data_f))
# add a column of 1 to X for bias term. ones((# of row, # or column))
X = np.hstack((np.ones((X.shape[0], 1)), X))

# Get y, y = [ 1.  1.  1. ... -1. -1. -1.]
len_m = len(data_m)
len_f = len(data_f)
y = np.hstack((np.array(np.ones(len_m)), np.array(-1.0 * np.ones(len_f))))

# theta hat = (X^T * X)^(pseudoinvers) * X^T * y.
theta_hat = np.dot(np.dot(np.linalg.pinv(np.dot(X.T,X)),X.T),y)
print(theta_hat)


# ---------------------------------------- (c: CVXPY(convex optimization)) ----------------------------------------
# Define variables
theta = cp.Variable(X.shape[1]) # shape[1] return num of column

# Define the objective (minimize like L2 norm. For L1, just use np.linalg.norm(y - (X @ theta), ord=1) or sum) 
objective = cp.Minimize(cp.sum_squares(y - (X @ theta))) # @ is for  matrix multiplication, but like matlab, it wait

# Define the problem
problem = cp.Problem(objective)

# Solve the problem
problem.solve()

# Extract the solution (theta_hat)
theta_hat_cvxpy = theta.value
print(theta_hat_cvxpy)


# ---------------------------------------- (d) ----------------------------------------
# See note. Gradient descent covered in Week4 P2 
# ∇E train = -d = 2 * x^T * x * theta - 2 * x^T * y 
# optimal step size, alpha = (y^T*x*d - theta^T*x^T*x*d) / ||x*d||^2


# ---------------------------------------- (e: gradient descent) ----------------------------------------
# gradient descent: x^(t+1) = x^t - Eta * [gradient of f(x^t)]. (covered in Week 4. x is "theta" in previous lectures)
# x is parameter to optimized, Eta is step size, and f() is loss function.

# initialize all elements
dimension = X.shape[1] # 3 here
max_iteration = 50000
loss = np.zeros(max_iteration) # [0. 0. ...... 0.]
theta = [0., 0., 0.] # Initializing theta with zeros 
XtX = np.dot(X.T, X) 
store_all_theta = np.zeros((dimension, max_iteration + 1)) # row, column

# gradient descent algo
for iter in range(max_iteration):
    # ∇E train = -d = 2 * x^T * x * theta - 2 * x^T * y 
    E_train = 2 * np.dot(XtX, theta) - 2 * np.dot(X.T, y)
    d = - E_train

    # alpha(step size), see note
    alpha = (np.dot(np.dot(y.T, X), d) - np.dot(np.dot(theta,XtX),d)) / np.sum((np.dot(X,d))**2)
    theta = theta + alpha * d # gradient descent (x^(t+1) = x^t - Eta * [gradient of f(x^t)])
    store_all_theta[:, iter + 1] = theta
    loss[iter] = np.linalg.norm(y - np.dot(X, theta))**2 / X.shape[0] # L2 by default. Squared L2 norm is often used in optimization problems 

print(theta)


# ---------------------------------------- (f) ----------------------------------------
# plt.semilogx(loss,'o',linewidth=8)
# plt.show()


# ---------------------------------------- (g) ----------------------------------------
# initialize all elements
beta = 0.9
dimension = X.shape[1] # 3 here
max_iteration = 50000
loss = np.zeros(max_iteration)
theta = [0., 0., 0.] 
XtX = np.dot(X.T, X) 
store_all_theta = np.zeros((dimension, max_iteration + 1))

# gradient descent algo
# ∇E train = -d = 2 x^T(x * theta - y)  (direction. Covered in Week 4. d = negative of gradient f(x^t))
iter = 0
E_train = 2 * np.dot(XtX, theta) - 2 * np.dot(X.T, y)
E_train_lastime = 2 * (np.dot(XtX, store_all_theta[:, iter]) - np.dot(X.T, y)) # changed
d = - (beta * E_train_lastime + (1 - beta) * E_train) # changed
alpha = (np.dot(np.dot(y.T, X), d) - np.dot(np.dot(theta,XtX),d)) / np.sum((np.dot(X,d))**2)
theta = theta + alpha * d # gradient descent (x^(t+1) = x^t - Eta * [gradient of f(x^t)])
store_all_theta[:, iter + 1] = theta
loss[iter] = np.linalg.norm(y - np.dot(X, theta))**2 / X.shape[0] # Squared L2 norm is often used in optimization problems 

for iter in range(1, max_iteration):
    # ∇E train = -d = 2 * x^T * x * theta - 2 * x^T * y 
    E_train = 2 * np.dot(XtX, theta) - 2 * np.dot(X.T, y)
    E_train_lastime = 2 * (np.dot(XtX, store_all_theta[:, iter - 1]) - np.dot(X.T, y)) # diff to iter 0, iter -1 here
    d = - (beta * E_train_lastime + (1 - beta) * E_train) # changed
    alpha = (np.dot(np.dot(y.T, X), d) - np.dot(np.dot(theta,XtX),d)) / np.sum((np.dot(X,d))**2)
    theta = theta + alpha * d # gradient descent (x^(t+1) = x^t - Eta * [gradient of f(x^t)])
    store_all_theta[:, iter + 1] = theta
    loss[iter] = np.linalg.norm(y - np.dot(X, theta))**2 / X.shape[0] # Squared L2 norm is often used in optimization problems 

print(theta)

# ---------------------------------------- (h) ----------------------------------------
# plt.semilogx(loss,'o',linewidth=8)
# plt.ylim(0.5, 1.0)  # Set the y-axis limits
# plt.show()


print("\nExercise 3")
# ---------------------------------------- Exercise 3: viualization and testing ----------------------------------------
# ---------------------------------------- (a) ----------------------------------------
# plt.scatter(data_m[:,0],data_m[:,1],c = 'b',alpha = 0.8,linewidth = 0.5) # data_f[:,0] is x1, data_m[:,1] is x2
# plt.scatter(data_f[:,0],data_f[:,1],c = 'r',alpha = 0.8,linewidth = 0.5) # x1 is bmi, x2 is stature
# plt.xlabel('bmi')
# plt.ylabel('stature_mm')
# plt.show()

plt.scatter(data_m[:,0],data_m[:,1],c = 'b',alpha = 0.8,linewidth = 0.5)
plt.scatter(data_f[:,0],data_f[:,1],c = 'r',alpha = 0.8,linewidth = 0.5)
plt.xlabel('bmi')
plt.ylabel('stature_mm')
x1 = np.linspace(1,9,100) # put x1 in x axis
x2 = (-theta[0] - theta[1] * x1) / theta[2] # put x2 in y axis. gθ(x) = θ0 + θ1x1 + θ2x2 = 0 for dicriminate function, that's why x2 is like that
plt.plot(x1,x2) 
plt.show()

# ---------------------------------------- (b) ----------------------------------------
# -------------------- (i) --------------------
# Type 1 error is the mistake of rejecting something that is actually true (False Positive)
femala_x2_predict =  (-theta[0] - theta[1] * data_f[:,0]) / theta[2] 
male_x2_predict =  (-theta[0] - theta[1] * data_m[:,0]) / theta[2]

false_alarm = 0
for i in range(len_f):
    if (data_f[i,1] > femala_x2_predict[i]):
        false_alarm = false_alarm + 1
false_alarm = false_alarm / len_f 
print("false_alarm =", false_alarm * 100, "%")

# -------------------- (ii) --------------------
# Type 2 error is the mistake of failing to reject something that is actually false (False Negative)
Miss = 0
for i in range(len_m):
    if (data_m[i,1] < male_x2_predict[i]):
        Miss = Miss + 1
Miss = Miss / len_m 
print("Miss =",Miss * 100,"%")

# -------------------- (iii) --------------------
TP = len_m - Miss * len_m 
FP = Miss * len_m
FN = false_alarm * len_f
precision = (TP)/(TP + FP) 
recall = TP/(TP + FN) 

print("precision =",precision * 100, "%, recall =", recall * 100,"%")


print("\nExercise 4")
# ---------------------------------------- Exercise 4: ----------------------------------------
# ---------------------------------------- (a) ----------------------------------------
lambd = np.arange(0.1,10,0.1)
Theta_values = []

for i in range(len(lambd)):
    THETA = cp.Variable(X.shape[1]) # initialize
    objective = cp.Minimize(cp.sum_squares(X @ THETA - y) + lambd[i] * cp.sum_squares(THETA))
    prob = cp.Problem(objective)
    
    result = prob.solve()
    Theta_values.append(THETA.value)

theta_norm_values = [] # this list stores the L2 norm of the parameter vector (theta) for each lambda value. 
residual_norm_values = [] # This list stores the L2 norm of the residuals for each lambda value.

for i in range(len(lambd)):
    theta_norm_values.append(np.linalg.norm(Theta_values[i]))
    residual_norm_values.append(np.linalg.norm(X @ Theta_values[i] - y))

plt.plot(theta_norm_values, residual_norm_values)
plt.show()
plt.plot(lambd, residual_norm_values)
plt.show()
plt.plot(lambd, theta_norm_values)
plt.show()

# ---------------------------------------- (b) ----------------------------------------
# see note


