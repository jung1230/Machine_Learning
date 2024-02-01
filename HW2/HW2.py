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
        row[2] = float(row[2]) / 100
        row[1] = float(row[1]) / 10
        modified_rows_female = np.row_stack((modified_rows_female, np.array(row, dtype=float)))
    
    # print the first 10 entries of female BMI
    # print the first 10 entries of female stature
    print(modified_rows_female[0:11, :])

csv_file.close()

# Reading csv file for male data
with open("male_train_data.csv", "r") as csv_file:
    reader = csv.reader(csv_file, delimiter=',')
    # normalize the number in male_stature_mm by dividing them with 1000
    # normalize that of male_bmi by dividing them with 10.
    
    counter = 0
    # Initialize modified_rows_male as an empty array
    modified_rows_male = np.empty((0, 3), dtype=float)
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
    print(modified_rows_male[0:11, :])

csv_file.close()






print("\nExercise 2")
# ---------------------------------------- Exercise 2: Build a Linear Classifier via Optimization ----------------------------------------
# ---------------------------------------- (a) ----------------------------------------
# theta hat = (A^T * A)^(-1) * A^T * y. This solution is for unique global minimum in terms of the rank of X
# Techniques when X^T * X is Not Invertible:
# 1. regularization: Ridge Regression(L2), LASSO(L1). from sklearn.linear_model import Lasso/Ridge
# 2. Pseudoinverse: numpy.linalg.pinv(). theta hat = (A^T * A)^(pseudoinvers) * A^T * y.


# ---------------------------------------- (b) ----------------------------------------
# Get A
# Round to decimals 3
data_m = np.around(modified_rows_male[1:,1:].astype(float), decimals = 3) 
data_f = np.around(modified_rows_female[1:,1:].astype(float), decimals = 3)
A = np.row_stack((data_m,data_f))
# add a column of 1 to the A for bias term
A = np.hstack((np.ones((A.shape[0], 1)), A))

# Get y
len_m = len(data_m)
len_f = len(data_f)
y = np.hstack((np.array(np.ones(len_m)), np.array(-1.0*np.ones(len_f))))

# get theta_hat with pseudoinverse
theta_hat = np.dot(np.dot(np.linalg.pinv(np.dot(A.T,A)),A.T),y)
print(theta_hat)

# ---------------------------------------- (c) ----------------------------------------
# Define variables
theta = cp.Variable(A.shape[1])

# Define the objective (minimize L2 norm) 
objective = cp.Minimize(cp.sum_squares(A @ theta - y)) # @ is for  matrix multiplication.

# Define the problem
problem = cp.Problem(objective)

# Solve the problem
problem.solve()

# Extract the solution (theta_hat)
theta_hat_cvxpy = theta.value

print(theta_hat_cvxpy)

# ---------------------------------------- (d) ----------------------------------------
