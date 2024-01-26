import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


# ------------------------- Exercise 1. -------------------------
# --------------- (a) ---------------
# Define the range of x values
x = np.linspace(-3, 3, 100)  # Adjust the number of points for a smoother curve

y = 1 / ((2 * np.pi) ** 0.5) * np.exp(-(x ** 2) / 2)

# Plot the curve
plt.plot(x, y, label='PDF of Standard Normal Distribution')
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.title('Standard Normal Distribution')
plt.legend() # show "PDF of Standard Normal Distribution"
plt.grid(True)
plt.savefig('Standard Normal Distribution')
plt.show()


# --------------- (b) ---------------
# ----- (1) -----
# Set the seed for reproducibility
np.random.seed(42)

# Draw 1000 random samples from N(0, 1)
samples = np.random.normal(0, 1, 1000)


# ----- (2) (3) (4) -----
# Create histograms
plt.figure(figsize=(12, 6))

# Create histogram 1
plt.subplot(1, 2, 1)

# Use density=True to normalize the histogram (make total area = 1)
plt.hist(samples, bins=4, color='blue', edgecolor='black', density=True)  
plt.title('Histogram with 4 Bins')
plt.title('Histogram with 4 Bins')

# Estimate mean and standard deviation using scipy.stats.norm.fit
mean, standard_dev = norm.fit(samples)  

# Report the estimated values
print("Estimated mean and standard deviation:", mean, standard_dev)

xmin, xmax = plt.xlim() # get the limit of the figure
x = np.linspace(xmin, xmax, 100)

# Fit a Gaussian curve and plot it
p = norm.pdf(x, mean, standard_dev)  # Use scipy.stats.norm.pdf to generate the PDF
plt.plot(x, p, 'k', linewidth=2)  # Plot the fitted Gaussian curve in black

# Create histogram 2
plt.subplot(1, 2, 2)

# Use density=True to normalize the histogram (make total area = 1)
plt.hist(samples, bins=1000, color='blue', edgecolor='black', density=True)  
plt.title('Histogram with 1000 Bins')
plt.plot(x, p, 'k', linewidth=2)  # Plot the fitted Gaussian curve in black

plt.show()
plt.close()

# ----- (5) -----
# Are the two histograms representative of your
# data’s distribution? How are they different in terms of data representation?

# Histogram with 4 Bins:
# This histogram has only 4 bins, which means the data is divided into four intervals or ranges.
# The fewer bins make it less detailed, and you might lose some information about the distribution.
# It provides a broad overview of the data but lacks the resolution to capture finer details.

# Histogram with 1000 Bins:
# This histogram has 1000 bins, providing a much more detailed representation of the data.
# The increased number of bins allows for a more granular view of the distribution, capturing variations in the data at a finer scale.
# It is useful for examining the distribution more closely, especially when the data has subtle features or variations.


# --------------- (c) ---------------
# ----- (1) -----
# Define the range of possible bin values (m)
m_values = np.linspace(1, 200, 200)
n = len(samples)  # Total number of samples

# Initialize arrays to store J values and corresponding h values
J_values = np.zeros_like(m_values)
h_values = (1 - 0) / m_values  # Calculate h for each m

# Loop over different values of m
for i, m in enumerate(m_values):
    # Create histogram with m bins
    hist, bin_edges = np.histogram(samples, bins = int(m), density=True)
    
    # Calculate empirical probabilities for each bin
    empirical_probabilities = hist * np.diff(bin_edges)
    
    # Calculate pj
    pj = np.sum(empirical_probabilities ** 2)
    
    # Calculate J for the current m
    J_values[i] = 2 / (h_values[i] * (n - 1)) - (n + 1) / (h_values[i] * (n - 1)) * pj

# Plot the risk J(h) with respect to m (x, y)
plt.plot(m_values, J_values)
plt.xlabel('Number of Bins (m)')
plt.ylabel('Risk (J)')
plt.title('Risk vs Number of Bins')
plt.show()

# ----- (2) -----
# Find the index of the minimum value in J_values
indexx = np.argmin(J_values) - 1
print(indexx)

# ----- (3) -----
# Draw 1000 random samples from N(0, 1)
samples = np.random.normal(0, 1, 1000)

# Use density=True to normalize the histogram (make total area = 1)
plt.hist(samples, bins = indexx, color='blue', edgecolor='black', density=True)  
plt.title(f'Histogram with {indexx} Bins')

# Estimate mean and standard deviation using scipy.stats.norm.fit
mean, standard_dev = norm.fit(samples)  

# Report the estimated values
print("Estimated mean and standard deviation:", mean, standard_dev)

xmin, xmax = plt.xlim() # get the limit of the figure
x = np.linspace(xmin, xmax, 100)

# Fit a Gaussian curve and plot it
p = norm.pdf(x, mean, standard_dev)  # Use scipy.stats.norm.pdf to generate the PDF
plt.plot(x, p, 'k', linewidth=2)  # Plot the fitted Gaussian curve in black

plt.show()
plt.close()




# ------------------------- Exercise 2. -------------------------

import numpy as np
import matplotlib.pyplot as plt



# (a)
# (i)
# To simplify the expression fX(x) for the given choices of µ and Σ:

# Given:
# X ∼ N(µ, Σ), where
# µ = [[2], [6]]
# Σ = [[2, 1], [1, 2]]

# The PDF of X is given by:
# fX(x) = 1 / (4 * pi^2 * |Σ|) * exp(-1/2 * (x - µ)ᵀ * Σ⁻¹ * (x - µ))

# Substitute the given values:
# fX(x) = 1 / (4 * pi^2 * |[[2, 1], [1, 2]]|) * exp(-1/2 * ([[x1], [x2]] - [[2], [6]])ᵀ * [[2, 1], [1, 2]]⁻¹ * ([[x1], [x2]] - [[2], [6]]))

# Calculate the determinant of Σ (|Σ|) and the inverse of Σ (Σ⁻¹), then substitute them back into the expression. The final simplified expression will provide the PDF for the given 2D Gaussian distribution.

# (ii)
# Define the parameters
mu = np.array([2, 6])
cov_matrix = np.array([[2, 1], [1, 2]])

# Define the function f_X(x)
def f_X(x):
    inv_cov_matrix = np.linalg.inv(cov_matrix)
    exponent_term = -0.5 * np.dot(np.dot((x - mu).T, inv_cov_matrix), (x - mu))
    normalization_term = 1 / (4 * np.pi**2)
    return normalization_term * np.exp(exponent_term)

# Create a grid of points for x ∈ [−1, 5] × [0, 10]
x_range = np.linspace(-1, 5, 100)
y_range = np.linspace(0, 10, 100)
X, Y = np.meshgrid(x_range, y_range)
Z = np.zeros_like(X)

# Evaluate f_X(x) for each point in the grid
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        Z[i, j] = f_X(np.array([X[i, j], Y[i, j]]))

# Plot the contour
plt.contour(X, Y, Z)
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Contour Plot of f_X(x)')
plt.show()




import numpy as np
import matplotlib.pyplot as plt

# (b)(i) Show that µY = b, and ΣY = A * A Transpose

# Given Y = AX + b
# Mean of Y: E[Y] = E[AX + b] = AE[X] + b = b (since E[X] = 0 for X ~ N(0, I))
# Covariance matrix of Y: E[(Y - µY)(Y - µY)^T] = E[(AX + b - b)(AX + b - b)^T] = E[AX * X^T * A^T] = A * E[X * X^T] * A^T = A * I * A^T = A * A^T

# (b)(ii) Show that ΣY is symmetric positive semi-definite

# A * A^T is always symmetric, and for any vector x, x^T * (A * A^T) * x = (A^T * x)^T * (A^T * x) >= 0

# (b)(iii) Condition for ΣY to be symmetric positive definite

# ΣY is symmetric positive definite if and only if A has full rank.

# (b)(iv) Determine A and b given µY and ΣY

# Given µY = [[2], [6]] and ΣY = [[2, 1], [1, 2]]
# Use eigen-decomposition: ΣY = P * D * P^T, where P is the matrix of eigenvectors and D is the diagonal matrix of eigenvalues
# A = P * sqrt(D), where sqrt(D) is the square root of D element-wise
# b = µY

# (c)(i) Verify with empirical example

# Draw 5000 random samples from the 2D standard normal distribution
data = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], 5000)

# Scatter plot of the data points
plt.scatter(data[:, 0], data[:, 1], alpha=0.5)
plt.title('Scatter Plot of 5000 Samples from N(0, I)')
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()

# (c)(ii) Apply the transformation

# Given A and b from (b)(iv)
A = np.dot(np.linalg.eigvals([[2, 1], [1, 2]]), np.linalg.eig([[2, 1], [1, 2]])[1])
b = [[2], [6]]

# Transform the data points
transformed_data = np.dot(A, data.T).T + b

# Scatter plot of the transformed data points
plt.scatter(transformed_data[:, 0], transformed_data[:, 1], alpha=0.5)
plt.title('Scatter Plot of Transformed Data Points')
plt.xlabel('Y1')
plt.ylabel('Y2')
plt.show()



import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.special import eval_legendre
from scipy.optimize import linprog

# ------------------------- Exercise 3. -------------------------
# --------------- (a) ---------------
# Set random seed for reproducibility
np.random.seed(42) 

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

#Theta = np.linalg.lstsq(A_legendre, y, rcond=None)[0] # do not compute (A^TA)^-1 Yourself!

# --------------- (c) ---------------
A_legendre =  np.column_stack((eval_legendre(0,x), eval_legendre(1,x), eval_legendre(2,x), eval_legendre(3,x), eval_legendre(4,x)))
# np.linalg.lstsq Return the least-squares solution to a linear matrix equatio
Theta = np.linalg.lstsq(A_legendre, y, rcond=None)[0]

y_predicted = A_legendre @ Theta
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

y_predicted = A_legendre @ Theta
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
# hstack: Stack arrays in sequence horizontally (column wise).
c_t = np.hstack((np.zeros(5), np.ones(50)))
# how column stack work???
X = np.column_stack((np.ones(50), eval_legendre(1,x), eval_legendre(2,x), eval_legendre(3,x), eval_legendre(4,x)))

# vstack: Stack arrays in sequence vertically (row wise).
# eye: Return a 2-D array with ones on the diagonal and zeros elsewhere. (input = row or column)
A = np.vstack((np.hstack((X, -np.eye(50))), np.hstack((-X, -np.eye(50)))))
b = np.hstack((y, -y))

res = linprog(c_t, A, b, bounds=(None, None), method="revised simplex")
theta = res.x[:5]  # Extract theta from the solution

# Calculate the predicted values using the obtained theta
y_predicted = X @ theta

# Scatter plot the data and overlay with the predicted curve
plt.scatter(x, y, label='Noisy Data')
plt.plot(x, y_true, label='True Model', color='red', linestyle='dashed')
plt.plot(x, y_predicted, label='Predicted Model', color='green')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()