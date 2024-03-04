import numpy as np 
import matplotlib.pyplot as plt
from PIL import Image
import cvxpy as cvx 

# ---------------------------------------- Exercise 2 ----------------------------------------
train_cat   = np.matrix(np.loadtxt('train_cat.txt',   delimiter = ','))
train_grass = np.matrix(np.loadtxt('train_grass.txt', delimiter = ','))
Y = plt.imread('cat_grass.jpg') / 255

# -------------------- (b: train by getting parameters) --------------------
row1, col1 = np.shape(train_cat)
u1 = np.zeros(row1)
# Loop through each row of the array and calculate the mean of the current row( SHOULD I CHANGE THIS TO COL?????????)
for i in range(row1):
    u1[i] = np.mean(train_cat[i,:])


row0, col0 = np.shape(train_grass)
u0 = np.zeros(row0)

# Loop through each row of the array and calculate the mean of the current row
for i in range(row0):
    u0[i] = np.mean(train_grass[i,:])

sigma1 = np.cov(train_cat)
sigma0 = np.cov(train_grass)

pi1 = col1 / (col0 + col1)
pi0 = col0 / (col0 + col1)

print("i:first two entry of u1 and u0")
print(u1[0], u1[1], u0[0], u0[1])

print("\nii: sigma1 and sigma2")
for i in range(2):
    print(sigma1[i][0],sigma1[i][1])
print("----------------------------------------")
for i in range(2):
    print(sigma0[i][0],sigma0[i][1])

print("\niii: pi1 and pi0")
print(pi1, pi0)

# -------------------- (c: prediction by using parameters) --------------------
M, N = Y.shape
prediction4Cat = np.zeros((M,N))

for i in range(M - 8):
    for j in range(N - 8):
        block = Y[i: i+8, j: j+8]
        block = np.reshape(block,(64,1)) # reshapes the array into a new shape specified by 64 row 1 column to match u1 and u0
        c1 = (-1/2) * (block - u1).T @ np.linalg.inv(sigma1) @ (block - u1) + np.log(pi1) - (1/2) * np.log(np.linalg.det(sigma1)) # equation from 2a
        c0 = (-1/2) * (block - u0).T @ np.linalg.inv(sigma0) @ (block - u0) + np.log(pi0) - (1/2) * np.log(np.linalg.det(sigma0))
        if np.sum(c1) > np.sum(c0): # compare the total cost (c1[0,0] > c0[0,0] works too?)
            prediction4Cat[i, j] = 1
        else:
            prediction4Cat[i, j] = 0

prediction4Cat = prediction4Cat * 255 # to make 1(cat) 255(white)
image = Image.fromarray(prediction4Cat)
image = image.convert('L')  # convert to grey scale
image.save('output4cat.png')
prediction4Cat = prediction4Cat / 255


# -------------------- (d) --------------------
# prediction_from_pic = plt.imread('output.png')# skip step 2 and run this code can be faster
# rowp , colp = prediction_from_pic.shape

rowp , colp = prediction4Cat.shape

truth = plt.imread('truth.png') 
MAE = 0
for i in range(rowp - 8):
    for j in range(colp - 8):
        MAE = MAE + np.abs(prediction4Cat[i, j] - truth[i, j])
        # MAE = MAE + np.abs(prediction_from_pic[i, j] - truth[i, j])
        
MAE = 1 / (rowp * colp) * MAE
print("MAE =", MAE)


# -------------------- (e) --------------------
tryPanda = plt.imread('Panda.jpg') / 255 # make it grey
M, N = tryPanda.shape
prediction4Panda = np.zeros((M,N))

for i in range(M - 8):
    for j in range(N - 8):

        block = tryPanda[i: i+8, j: j+8]
        block = np.reshape(block,(64,1)) # reshapes the array into a new shape specified by 64 row 1 column to match u1 and u0
        c1 = (-1/2) * (block - u1).T @ np.linalg.inv(sigma1) @ (block - u1) + np.log(pi1) - (1/2) * np.log(np.linalg.det(sigma1)) # equation from 2a
        c0 = (-1/2) * (block - u0).T @ np.linalg.inv(sigma0) @ (block - u0) + np.log(pi0) - (1/2) * np.log(np.linalg.det(sigma0))
        if np.sum(c1) > np.sum(c0): # compare the total cost (c1[0,0] > c0[0,0] works too?)
            prediction4Panda[i, j] = 1
        else:
            prediction4Panda[i, j] = 0

prediction4Panda = prediction4Panda * 255 # to make 1(cat) 255(white)
image = Image.fromarray(prediction4Panda)
image = image.convert('L')  # convert to grey scale
image.save('output4Panda.png')
# it perform badly! I think the reason is it can't recognize the look of panda
