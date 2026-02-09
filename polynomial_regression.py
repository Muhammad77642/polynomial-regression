import numpy as np
import matplotlib.pyplot as plt

x = np.loadtxt("Ice_cream selling data.csv",delimiter=",",usecols=0,skiprows=1)
y = np.loadtxt("Ice_cream selling data.csv",delimiter=",",usecols=1,skiprows=1)
#scalable degree
def polynomial_features(x, degree):
    return np.column_stack([x**i for i in range(1, degree+1)])
# initializing curve
x_powered = polynomial_features(x, 2)

# initializing curve
# w,b
w = np.zeros(2)
b = 0
# ITERATIONS
iterations = 5000
# Learning rate
alpha = 0.01
# Normalizing data
def normalization(x):
    mu = np.mean(x,axis=0)
    sigma = np.std(x,axis=0)
    x_scaled = (x-mu) / sigma
    return x_scaled
x_normalized = normalization(x_powered)

# Computing cost function
def compute_cost(x,y,w,b):
    m = len(x)
    f_wb = x @ w + b
    error = (f_wb - y)**2
    cost = np.sum(error) / (2*m)
    return cost
# computing the gradient (Derivative)
def compute_gradient(x,y,w,b):
    m = len(x)
    f_wb = x @ w + b
    error = (f_wb - y)
    dj_dw = (x.T @ error) / m
    dj_db = np.sum(error) / m
    return dj_dw,dj_db
# Updating w,b

for i in range (iterations):
    dj_dw,dj_db = compute_gradient(x_normalized,y,w,b)
    w -= (alpha * dj_dw)
    b -= (alpha * dj_db)
    # tracking cost
    if i % 1000 == 0 :
        print(compute_cost(x_normalized,y,w,b))
print(f"optimal w :{w}, optimal b : {b} ")
#  Computing new Y
f_new = x_normalized @ w +  b
# visualization
x_line = np.linspace(min(x), max(x), 200)

x_poly = polynomial_features(x_line, 2)
x_poly_norm = (x_poly - np.mean(x_powered,axis=0)) / np.std(x_powered,axis=0)

y_line = x_poly_norm @ w + b

plt.scatter(x,y)
plt.plot(x_line,y_line)
plt.show()

# Metrics
def RMSE(y_orirgin,y_new):
    rmse = np.sqrt(np.mean((y_orirgin - y_new ) ** 2))
    return rmse
print(RMSE(y,f_new))



