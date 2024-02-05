import matplotlib.pyplot as plt
import numpy as np
from scipy.special import eval_legendre
from scipy.optimize import linprog


import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

samples = np.random.normal(0, 1, 1000)
x = np.linspace(-3, 3, 100) 


# Define the range of possible bin values (m)
m_values = np.linspace(1, 200, 200)

# Total number of samples
n = len(samples)  

# Initialize arrays to store J values
J_values = np.zeros_like(m_values)

#  h = (max data value−min data value)/m
h_values = (1 - 0) / m_values  # Calculate h for each m

# Loop over different values of m (i = index, m = value)
for i, m in enumerate(m_values):
    # Create histogram with m bins. 
    # hist = counts of samples in each bin of the histogram, 
    # bin_edges = the edges of the bins
    hist, bin_edges = np.histogram(samples, bins = int(m), density=True)
    
    # Calculate empirical probabilities for each bin
    # np.diff(bin_edges) = the width of each bin.
    # provides an estimate of the area under the histogram bar for that bin. In a probability density histogram (when density=True` is used), this area is proportional to the probability of a random sample falling within that bin.
    empirical_probabilities = hist * np.diff(bin_edges)
    
    # Calculate pj (empirical probability of a sample falling into each bin)
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
indexx = np.argmin(J_values) 
print(indexx)

# ----- (3) -----
# Draw 1000 random samples from N(0, 1)
samples = np.random.normal(0, 1, 1000)

# Use density=True to normalize the histogram (make total area = 1)
plt.hist(samples, bins = indexx, color='blue', edgecolor='black', density=True)  
plt.title(f'Histogram with {indexx} Bins')

# Fit a Gaussian curve and plot it
p = norm.pdf(x, 0, 1)  # Use scipy.stats.norm.pdf to generate the PDF
plt.plot(x, p, 'k', linewidth=2)  # Plot the fitted Gaussian curve in black

plt.show()
plt.close()