#!/usr/bin/envpython
# %% imports
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
get_ipython().run_line_magic('matplotlib', 'inline')


# %% [markdown]
# ## Defining the cost function
# $$cost(\theta_{1}, \theta_{2}) = \theta_{1}^2 + \theta_{2}^2$$
# $$dcost(\theta_{1}, \theta_{2}) = 2\,(\theta_{1} + \theta_{2})$$

# %% definining the cost function "cost" and its gradient "dcost" with respect to θ1 et θ2
def cost(theta):
    return theta[0]**2 + theta[1]**2

def dcost(theta):
    return 2*theta

# %% some boilerplate for ploting the surface of the function
theta1_axis = np.linspace(-2, 2, 41)
theta2_axis = np.linspace(-2, 2, 41)
theta1_axis , theta2_axis = np.meshgrid(theta1_axis, theta2_axis)
cost_plot = np.zeros((len(theta1_axis), len(theta2_axis)))
for j in range(len(theta1_axis)):
    for k in range(len(theta2_axis)):
        cost_plot[j, k] = cost(np.array([theta1_axis[j, k] , theta2_axis[j, k]]))


# get_ipython().run_line_magic('matplotlib', 'inline')


# %% defining the plot functions
fig = plt.figure(figsize=(22, 18))
ax = fig.add_subplot(1, 1, 1, projection='3d', azim=-110, elev=22)
ax.plot_wireframe(theta1_axis, theta2_axis, cost_plot)
ax.set_xlabel('theta1', fontsize=10)
ax.set_ylabel('theta2', fontsize=10)
ax.set_zlabel('cost function', fontsize=8)
plt.show()


# %% defining the plot functions
def plot_solution(theta_vec, figure_nmbr=1):
    fig = plt.figure(figure_nmbr, figsize=(22, 18))
    n = theta_vec.shape[1]
    ax1 = fig.add_subplot(1, 2, 1, projection='3d', azim=-110, elev=22)
    ax1.plot_wireframe(theta1_axis, theta2_axis, cost_plot)
    ax1.set_xlabel('theta1', fontsize=10)
    ax1.set_ylabel('theta2', fontsize=10)
    ax1.set_zlabel('cost function', fontsize=8)
    ax1.scatter(theta_vec[0, :], theta_vec[1, :], cost(theta_vec), c='r', s=40)

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_xlim(-1, n)
    ax2.set_ylim(np.min(theta_vec)-0.1, np.max(theta_vec)+0.1)
    ax2.scatter(np.linspace(0, n-1, n), theta_vec[0, :], c = 'b', label='theta1')
    ax2.scatter(np.linspace(0, n-1, n), theta_vec[1, :], c = 'c', label='theta2')
    fig.tight_layout()
    plt.show()


# %% [markdown]
# ## Stochastic Gradient Descent

# %% Stochastic Gradient Descent
def SGD(theta0, lr=0.01, tol=1e-6, max_i=2500):
    theta_vec, cost_vec = np.zeros((2, max_i+1)), np.zeros((max_i+1))
    theta_vec[:, 0], cost_vec[0] = theta0, cost(theta0)
    for i in range(0, max_i):
        theta_vec[:, i+1] = theta_vec[:, i] - lr * dcost(theta_vec[:, i])
        cost_vec[i+1] = cost(theta_vec[:, i+1])
        if abs(cost_vec[i+1]-cost_vec[i]) < tol:
            break
    return theta_vec[:, :i+2]


# %% [markdown]
# ## Stochastic Gradient Descent with Momentum

# %% Momentum SGD
def MomenSGD(theta0, lr=0.01, md=0.9, tol=1e-6, max_i=2500):
    theta_vec, cost_vec = np.zeros((2, max_i+1)), np.zeros((max_i+1))
    v = np.zeros((2))
    theta_vec[:, 0], cost_vec[0] = theta0, cost(theta0)
    for i in range(0, max_i):
        v = md*v - lr*dcost(theta_vec[:, i])
        theta_vec[:, i+1] = theta_vec[:, i] + v
        cost_vec[i+1] = cost(theta_vec[:, i+1])
        if abs(cost_vec[i+1]-cost_vec[i]) < tol:
            break
    return theta_vec[:, :i+2]


# %% Nasterov Accelerated Gradient
def Nag(theta0, lr=0.01, md=0.9, tol=1e-6, max_i=2500):
    theta_vec, cost_vec = np.zeros((2, max_i+1)), np.zeros((max_i+1))
    theta_vec[:, 0], cost_vec[0] = theta0, cost(theta0)
    v = np.zeros((2))
    for i in range(0, max_i):
        v = md*v - lr*dcost(theta_vec[:, i]+md*v)
        theta_vec[:, i+1] = theta_vec[:, i] + v
        cost_vec[i+1] = cost(theta_vec[:, i+1])
        if abs(cost_vec[i+1]-cost_vec[i]) < tol:
            break
    return theta_vec[:, :i+2]

# %% Adapative Subgradient
def Adagrad(theta0, lr=0.01, tol=1e-6, max_i=2500):
    theta_vec, cost_vec = np.zeros((2, max_i+1)), np.zeros(max_i+1)
    theta_vec[:, 0], cost_vec[0] = theta0, cost(theta0)
    s = np.zeros((2))
    for i in range(0, max_i):
        dtheta = dcost(theta_vec[:, i])
        s += dtheta**2
        theta_vec[:, i+1] = theta_vec[:, i] - dtheta * lr / np.sqrt(s + 1e-8)
        cost_vec[i+1] = cost(theta_vec[:, i+1])
        if abs(cost_vec[i+1]-cost_vec[i]) < tol:
            break
    return theta_vec[:, :i+2]

# %% RMS Prop
def RMS(theta0, lr=0.01, wd=0.9, tol=1e-6, max_i=2500):
    theta_vec, cost_vec  = np.zeros((2, max_i+1)), np.zeros(max_i+1)
    theta_vec[:, 0], cost_vec[0], s = theta0, cost(theta0), 0
    s = np.zeros((2))
    for i in range(0, max_i):
        s = wd * s + (1-wd) * dcost(theta_vec[:, i])**2
        theta_vec[:, i+1] = theta_vec[:, i] - dcost(theta_vec[:, i]) * lr / np.sqrt(s+1e-6)
        cost_vec[i+1] = cost(theta_vec[:, i+1])
        if abs(cost_vec[i+1]-cost_vec[i]) < tol:
            break
    return theta_vec[:, :i+2]

# %% AdaDelta
def AdaDelta(theta0, lr=0.01, wd=0.9, tol=1e-6, max_i=2500):
    theta_vec, cost_vec  = np.zeros((2, max_i+1)), np.zeros(max_i+1)
    theta_vec[:, 0], cost_vec[0] = theta0, cost(theta0)
    Sg = np.ones((2))
    Sd = np.ones((2))
    for i in range(0, max_i):
        Sg = wd * Sg + (1-wd) * dcost(theta_vec[:, i])**2
        dtheta = -dcost(theta_vec[:, i]) * lr * np.sqrt(Sd + 1e-6) / np.sqrt(Sg + 1e-6)
        Sd = wd * Sd + (1-wd) * dtheta**2
        theta_vec[:, i+1] = theta_vec[:, i] + dtheta
        cost_vec[i+1] = cost(theta_vec[:, i+1])
        if abs(cost_vec[i+1]-cost_vec[i]) < tol:
            break
    return theta_vec[:, :i+2]

# %% Adam
def Adam(theta0, lr=0.01, b1=0.9, b2=0.999, dc=(1-1e-8), tol=1e-6, max_i=2500):
    theta_vec, cost_vec  = np.zeros((2, max_i+1)), np.zeros(max_i+1)
    theta_vec[:, 0], cost_vec[0] = theta0, cost(theta0)
    S, M = np.zeros((2,)), np.zeros((2,))
    b1t = b1
    for i in range(0, max_i):
        b1t *= dc
        grad = dcost(theta_vec[:, i])
        M = b1t * M + (1 - b1t) * grad
        S = b2 * S + (1 - b2) * grad**2
        m = M/(1-b1**(i+1))
        s = S/(1-b2**(i+1))
        theta_vec[:, i+1] = theta_vec[:, i] - lr * m / np.sqrt(s + 1e-8)
        cost_vec[i+1] = cost(theta_vec[:, i+1])
        if abs(cost_vec[i+1]-cost_vec[i]) < tol:
            break
    return theta_vec[:, :i+2]


# %% tune function
def tune_opt(Opt, theta0, lrmin, lrmax):
    best_iter, best_lr = 1000, None
    for lr in np.logspace(np.log10(lrmin), np.log10(lrmax), 100):
        theta_vec = Opt(theta0, lr=lr)
        nmbr_iter = theta_vec.shape[1]
        if cost(theta_vec[:,-1]) < 1e-6 and nmbr_iter < best_iter:
            best_lr, best_iter = lr, nmbr_iter
    return best_lr


# %% tune function
t01 = np.array([-2, -2])

# %% tune function
plot_solution(SGD(t01, lr=0.01), 1)

# %% tune function
plot_solution(MomenSGD(t01, lr=0.01), 2)

# %% tune function
plot_solution(Nag(t01, lr=0.01), 3)

# %% tune function
plot_solution(Adagrad(t01, lr=0.4), 4)

# %% tune function
plot_solution(RMS(t01, lr=0.1), 5)

# %% tune function
plot_solution(AdaDelta(t01, lr=0.2), 6)

# %% tune function
plot_solution(Adam(t01, lr=0.1), 7)
