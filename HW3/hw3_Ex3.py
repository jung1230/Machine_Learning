import numpy as np 
import matplotlib.pyplot as plt
from PIL import Image
import cvxpy as cvx 

# ---------------------------------------- Exercise 3 ----------------------------------------
# -------------------- (a) --------------------
# threshold constand(τ) = pi0 / pi1 = 0.828650711064863 / 0.0.171349288935137 = 4.836

# -------------------- (b) --------------------
# extra
prediction4Cat = plt.imread('output4cat.png')# skip step 2 and run this code can be faster
rowp , colp = prediction4Cat.shape
truth = plt.imread('truth.png') 
Y = plt.imread('cat_grass.jpg') / 255

true_positive = 0 # true for actual, Positive for predicted
total_positive4truth = np.count_nonzero(truth)
false_positive = 0 # false for actual, positive for predicted
total_negative4truth = len(truth.ravel()) - total_positive4truth # ravel: Return a contiguous flattened array.

# left part of comparison 
train_cat   = np.matrix(np.loadtxt('train_cat.txt',   delimiter = ',')) # extra
train_grass = np.matrix(np.loadtxt('train_grass.txt', delimiter = ',')) # extra
d = 64 # train_cat.shape = (64, 1976)

sigma1 = np.cov(train_cat) # extra
sigma0 = np.cov(train_grass) # extra
abs_Sigma1 = np.linalg.det(sigma1)
abs_Sigma0 = np.linalg.det(sigma0)
inv_Sigma1 = np.linalg.inv(sigma1)
inv_Sigma0 = np.linalg.inv(sigma0)

# extra
row1, col1 = np.shape(train_cat)
u1 = np.zeros(row1)
for i in range(row1):
    u1[i] = np.mean(train_cat[i,:])

# extra
row0, col0 = np.shape(train_grass)
u0 = np.zeros(row0)
for i in range(row0):
    u0[i] = np.mean(train_grass[i,:])

div = np.zeros((rowp,colp))
for i in range(rowp - 8):
    for j in range(colp - 8):
        block = Y[i: i+8, j: j+8]
        block = np.reshape(block,(64,1))
        PofXgivenC1 = 1/(((2*np.pi)**(d/2)) * ((abs_Sigma1)**(0.5))) * np.exp(-0.5 * (block - u1).T @ inv_Sigma1 @ (block - u1))
        PofXgivenC0 = 1/(((2*np.pi)**(d/2)) * ((abs_Sigma0)**(0.5))) * np.exp(-0.5 * (block - u0).T @ inv_Sigma0 @ (block - u0))
        div[i,j] = np.log(PofXgivenC1[0,0] / PofXgivenC0[0,0]) # !!!! try:change to np.sum and mean!!!!

n_tau = 50
PF = np.zeros(n_tau)
PD = np.zeros(n_tau)
tauset = np.linspace(-200,200,n_tau)
for k in range(n_tau):
    tau = (tauset[k])
    prediction = np.zeros(Y.shape)
    true_positive = 0; 
    false_positive = 0; 
    prediction = div > tau
    for i in range(rowp - 8):
        for j in range(colp - 8):
            if (prediction[i][j]==1) and (truth[i][j]>=0.5): true_positive  += 1
            if (prediction[i][j]==1) and (truth[i][j]<=0.5): false_positive += 1


    PD[k] = true_positive / total_positive4truth
    PF[k] = false_positive / total_negative4truth

plt.figure()
plt.plot(PF,PD)
plt.xlabel('PF')
plt.ylabel('PD')
plt.title('ROC curve')
plt.xlim(0, 1)  
plt.ylim(0, 1) 

# plt.show() 

        
# -------------------- (c) --------------------
tau = 4.836
prediction = np.zeros(Y.shape)
true_positive = 0; 
false_positive = 0; 
prediction = div > tau
for i in range(rowp - 8):
    for j in range(colp - 8):
        if (prediction[i][j]==1) and (truth[i][j]>=0.5): true_positive  += 1
        if (prediction[i][j]==1) and (truth[i][j]<=0.5): false_positive += 1


plt.scatter(false_positive / total_negative4truth, true_positive / total_positive4truth, s=100, c='red', label='Specific Point')  
plt.show() 

# -------------------- (d) --------------------
X1 = train_cat.T
X0 = train_grass.T
A = np.vstack((X1, X0))
Ones4X1 = np.ones((col1,1))
Ones4X0 = np.ones((col0,1)) 
b = np.vstack((Ones4X1,-1 * Ones4X0))

theta = cvx.Variable((d,1))
objective = cvx.Minimize(cvx.sum_squares(A @ theta - b)) # sum_squares for 2-norm, norm for 1-norm
prob = cvx.Problem(objective)
prob.solve()
theta_hat = theta.value


thetaX = np.zeros((rowp,colp))
for i in range(rowp - 8):
    for j in range(colp - 8):
        block = Y[i: i+8, j: j+8]
        block = np.reshape(block,(64,1))
        thetaX[i,j] = theta_hat.T @ block

n_tau = 50
PF = np.zeros(n_tau)
PD = np.zeros(n_tau)
tauset = np.linspace(-1.2, -0.1, n_tau)
for k in range(n_tau):
    tau = (tauset[k])
    prediction = np.zeros(Y.shape)
    true_positive = 0; 
    false_positive = 0; 
    prediction = thetaX > tau
    for i in range(rowp - 8):
        for j in range(colp - 8):
            if (prediction[i][j]==1) and (truth[i][j]>=0.5): true_positive  += 1
            if (prediction[i][j]==1) and (truth[i][j]<=0.5): false_positive += 1


    PD[k] = true_positive / total_positive4truth
    PF[k] = false_positive / total_negative4truth

plt.figure()
plt.plot(PF,PD)
plt.xlabel('PF')
plt.ylabel('PD')
plt.title('ROC curve')
plt.xlim(0, 1)  
plt.ylim(0, 1) 
plt.show() 