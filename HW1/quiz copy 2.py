import matplotlib.pyplot as plt
import numpy as np
from scipy.special import eval_legendre
from scipy.optimize import linprog


# ----------- (a) -----------
# (i)
# See note

# (ii)
# Define the function Fx
def Fx(X1, X2):
    # x = [[x1], [x2]], x ∈ [−1, 5]×[0, 10]
    return 1 / (2 * (3)**0.5 * np.pi) * np.exp((-X1**2 + X1*X2 - X2**2 ) / 3)

# Generate a grid of points within the specified range
x1_values = np.linspace(-5, 5, 100)
x2_values = np.linspace(-5, 5, 100)
X1, X2 = np.meshgrid(x1_values, x2_values)

Z = Fx(X1, X2)

# Plot the contours using matplotlib.pyplot.contour
plt.contour(X1, X2, Z) 
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Contour Plot of Fx')
plt.colorbar(label='Fx')
plt.show()