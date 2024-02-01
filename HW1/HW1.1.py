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
np.random.seed(4)

# Draw 1000 random samples from N(0, 1)
samples = np.random.normal(0, 1, 1000)


# ----- (2) (3) (4) -----
# set size of the histograms
plt.figure(figsize=(12, 6))

# Create histogram 1
plt.subplot(1, 2, 1)

# Use density=True to normalize the histogram (make total area = 1)
plt.hist(samples, bins=4, color='blue', edgecolor='black', density=True)  
plt.title('Histogram with 4 Bins')

# Estimate mean and standard deviation using scipy.stats.norm.fit
mean, standard_dev = norm.fit(samples)  

# Report the estimated values
print("Estimated mean and standard deviation:", mean, standard_dev)

xmin, xmax = plt.xlim() # get the limit of the figure
x = np.linspace(xmin, xmax, 100)

# Use scipy.stats.norm.pdf to generate the PDF
p = norm.pdf(x, mean, standard_dev)  

# Plot the fitted Gaussian curve in black
plt.plot(x, p, 'k', linewidth=2)  

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


# --------------- (c) optimal bin ---------------
# ----- (1) -----
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

