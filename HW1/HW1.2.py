# ------------------------- Exercise 2. -------------------------
import numpy as np
import matplotlib.pyplot as plt
import math
# ----------- (a) -----------
# (i)
# See note

# (ii)
# Define the function Fx
def Fx(X1, X2):
    # x = [[x1], [x2]], x ∈ [−1, 5]×[0, 10]
    return 1 / (2 * (3)**0.5 * np.pi) * np.exp((-X1**2 - 2*X1 + X1*X2 - X2**2 - 28 + 10*X2) / 3)

# Generate a grid of points within the specified range
x1_values = np.linspace(-1, 5, 100)
x2_values = np.linspace(0, 10, 100)
X1, X2 = np.meshgrid(x1_values, x2_values)
Z = Fx(X1, X2)

# Plot the contours using matplotlib.pyplot.contour
plt.contour(X1, X2, Z) 
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Contour Plot of Fx')
plt.colorbar(label='Fx')
plt.show()


# ----------- (b) -----------
# see note

# ----------- (c) -----------
# ----- (i) -----
# Draw 5000 random samples from the 2D standard normal distribution
mean = [0,0]
cov = [[1, 0],[0, 1]]
x1 = np.random.multivariate_normal(mean, cov, 5000)
plt.scatter(x1[:,0], x1[:,1])
plt.show()

mean = [2,6]
cov = [[2, 1],[1, 2]]
x2 = np.random.multivariate_normal(mean, cov, 5000)
plt.scatter(x2[:,0], x2[:,1])
plt.show()

# ----- (ii) -----
# L is a 1D array containing the eigenvalues of the matrix.
# U is a 2D array where each column represents an eigenvector corresponding to an eigenvalue in L.
L, U = np.linalg.eig([[2, 1], [1, 2]])
A = np.dot(np.dot(U,np.diag(L**0.5)),U.T)

# remember to do transpose for matrix's dot product's multiplicand everytime???
transformed_data = np.dot(A, x1.T).T + mean
plt.scatter(transformed_data[:, 0], transformed_data[:, 1])
plt.title('Scatter Plot of Transformed Data Points')
plt.show()