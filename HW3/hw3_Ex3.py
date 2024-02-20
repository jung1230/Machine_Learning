import numpy as np 
import matplotlib.pyplot as plt
from PIL import Image


# ---------------------------------------- Exercise 3 ----------------------------------------
# -------------------- (a) --------------------
# threshold constand(τ) = pi0 / pi1 = 0.828650711064863 / 0.0.171349288935137 = 4.836

# -------------------- (b) --------------------
true_positive = 0 # true for actual, Positive for predicted
total_positive4truth = 0
false_positive = 0 # false for actual, positive for predicted
total_negative4truth = 0

prediction4Cat = plt.imread('output4cat.png')# skip step 2 and run this code can be faster
rowp , colp = prediction4Cat.shape

truth = plt.imread('truth.png') 



for i in range(rowp - 8):
    for j in range(colp - 8):
        if(truth[i, j] == 1): # if it is white(cat)
            total_positive4truth += 1
        else:
            total_negative4truth += 1
        
        if(truth[i, j] == 1 and prediction_from_pic[i, j] == 1):
            true_positive += 1
        if(truth[i, j] == 0 and prediction_from_pic[i, j] == 1):
            false_positive += 1
        
    