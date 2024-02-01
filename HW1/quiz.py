import matplotlib.pyplot as plt
import numpy as np
from scipy.special import eval_legendre
from scipy.optimize import linprog


import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


# --------------- (b) ---------------
# ----- (1) -----
# Set the seed for reproducibility
np.random.seed(4)

# Draw 1000 random samples from N(0, 1)
samples = np.random.normal(0, 1, 1000)

# Use density=True to normalize the histogram (make total area = 1)
plt.hist(samples, bins=15, color='blue', edgecolor='black', density=True)  

xmin, xmax = plt.xlim() # get the limit of the figure
x = np.linspace(xmin, xmax, 100)

# Use scipy.stats.norm.pdf to generate the PDF
p = norm.pdf(x, 0, 1)  

# Plot the fitted Gaussian curve in black
plt.plot(x, p, 'k', linewidth=2)  

plt.show()
plt.close()