import numpy as np
import pandas as pd
import torch
import random
from synthetic import simulate_var,simulate_lorenz_96
from sklearn import preprocessing
from copy import deepcopy
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve, roc_auc_score

fix_seed = 230526
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

device = torch.device('cuda')


# Based on https://github.com/smkalami/lotka-volterra-in-python
import numpy as np


class MultiLotkaVolterra:
    def __init__(self, p=10, d=2, alpha=1.2, beta=0.2, gamma=1.1, delta=0.05, sigma=0.1):
        """
        Dynamical multi-species Lotka--Volterra system. The original two-species Lotka--Volterra is a special case
        with p = 1 , d = 1.

        @param p: number of predator/prey species. Total number of variables is 2*p.
        @param d: number of GC parents per variable.
        @param alpha: strength of interaction of a prey species with itself.
        @param beta: strength of predator -> prey interaction.
        @param gamma: strength of interaction of a predator species with itself.
        @param delta: strength of prey -> predator interaction.
        @param sigma: scale parameter for the noise.
        """

        assert p >= d and p % d == 0

        self.p = p
        self.d = d

        # Coupling strengths
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.sigma = sigma

    def simulate(self, t: int, dt=0.01, downsample_factor=10, seed=None):
        if seed is not None:
            np.random.seed(seed)
        xs_0 = np.random.uniform(10, 100, size=(self.p, ))
        ys_0 = np.random.uniform(10, 100, size=(self.p, ))

        ts = np.arange(t) * dt

        # Simulation Loop
        xs = np.zeros((t, self.p))
        ys = np.zeros((t, self.p))
        xs[0, :] = xs_0
        ys[0, :] = ys_0
        for k in range(t - 1):
            xs[k + 1, :], ys[k + 1, :] = self.next(xs[k, :], ys[k, :], dt)

        causal_struct = np.zeros((self.p * 2, self.p * 2))
        signed_causal_struct = np.zeros((self.p * 2, self.p * 2))
        for j in range(self.p):
            # Self causation
            causal_struct[j, j] = 1
            causal_struct[j + self.p, j + self.p] = 1

            signed_causal_struct[j, j] = +1
            signed_causal_struct[j + self.p, j + self.p] = -1

            # Predator-prey relationships
            causal_struct[j, int(np.floor((j + self.d) / self.d) * self.d - 1 + self.p - self.d + 1):int(np.floor((j + self.d) / self.d) * self.d + self.p)] = 1
            causal_struct[j + self.p, int(np.floor((j + self.d) / self.d) * self.d - self.d + 1 - 1):int(np.floor((j + self.d) / self.d) * self.d)] = 1

            signed_causal_struct[j, int(np.floor((j + self.d) / self.d) * self.d - 1 + self.p - self.d + 1):int(np.floor((j + self.d) / self.d) * self.d + self.p)] = -1
            signed_causal_struct[j + self.p, int(np.floor((j + self.d) / self.d) * self.d - self.d + 1 - 1):int(np.floor((j + self.d) / self.d) * self.d)] = +1

        return [np.concatenate((xs[::downsample_factor, :], ys[::downsample_factor, :]), 1)], causal_struct, signed_causal_struct

    # Dynamics
    # State transitions using the Runge-Kutta method
    def next(self, x, y, dt):
        xdot1, ydot1 = self.f(x, y)
        xdot2, ydot2 = self.f(x + xdot1 * dt / 2, y + ydot1 * dt / 2)
        xdot3, ydot3 = self.f(x + xdot2 * dt / 2, y + ydot2 * dt / 2)
        xdot4, ydot4 = self.f(x + xdot3 * dt, y + ydot3 * dt)
        # Add noise to simulations
        xnew = x + (xdot1 + 2 * xdot2 + 2 * xdot3 + xdot4) * dt / 6 + \
               np.random.normal(scale=self.sigma, size=(self.p, ))
        ynew = y + (ydot1 + 2 * ydot2 + 2 * ydot3 + ydot4) * dt / 6 + \
               np.random.normal(scale=self.sigma, size=(self.p, ))
        # Clip from below to prevent populations from becoming negative
        return np.maximum(xnew, 0), np.maximum(ynew, 0)

    def f(self, x, y):
        xdot = np.zeros((self.p, ))
        ydot = np.zeros((self.p, ))

        for j in range(self.p):
            y_Nxj = y[int(np.floor((j + self.d) / self.d) * self.d - self.d + 1 - 1):int(np.floor((j + self.d) /
                                                                                                  self.d) * self.d)]
            x_Nyj = x[int(np.floor((j + self.d) / self.d) * self.d - self.d + 1 - 1):int(np.floor((j + self.d) /
                                                                                                  self.d) * self.d)]
            xdot[j] = self.alpha * x[j] - self.beta * x[j] * np.sum(y_Nxj) - self.alpha * (x[j] / 200) ** 2
            ydot[j] = self.delta * np.sum(x_Nyj) * y[j] - self.gamma * y[j]
        return xdot, ydot



def generate_mixed_var(p=10,typeflag=[0, 0, 1, 1, 1, 1, 1, 1, 1, 1],T=1000,lag=3,length_per_batch=50,device = torch.device('cuda')) -> object:
    X_np, beta, GC = simulate_var(p=p, T=T, lag=lag)
    min_max_scaler = preprocessing.MinMaxScaler()
    Y = min_max_scaler.fit_transform(X_np)
    X_pre = torch.tensor(Y[np.newaxis], dtype=torch.float32).reshape(-1, length_per_batch, p).numpy()

    # instance norm !!!
    X_real = np.zeros_like(X_pre)
    for i in range(X_real.shape[0]):
        for j in range(X_real.shape[-1]):
            instance = X_pre[i, :, j]
            instance_norm = (instance - np.min(instance)) / (np.max(instance) - np.min(instance))
            X_real[i, :, j] = instance_norm
    X_real = torch.tensor(X_real, dtype=torch.float32, device=device)
    X = deepcopy(X_real)

    for i in range(len(typeflag)):
        if typeflag[i] == 0:
            X[:, :, i] = X[:, :, i] > 0.5

    return X_real, X, GC


def generate_mixed_lorenz_96(p=10, typeflag=[0, 0, 1, 1, 1, 1, 1, 1, 1, 1], T=1000, F=5, length_per_batch=50,
                       device=torch.device('cuda')):

    X_np, GC = simulate_lorenz_96(p=p, F=F, T=T)
    min_max_scaler = preprocessing.MinMaxScaler()
    Y = min_max_scaler.fit_transform(X_np)
    X_pre = torch.tensor(Y[np.newaxis], dtype=torch.float32).reshape(-1, length_per_batch, p).numpy()

    # instance norm !!!
    X_real = np.zeros_like(X_pre)
    for i in range(X_real.shape[0]):
        for j in range(X_real.shape[-1]):
            instance = X_pre[i, :, j]
            instance_norm = (instance - np.min(instance)) / (np.max(instance) - np.min(instance))
            X_real[i, :, j] = instance_norm
    X_real = torch.tensor(X_real, dtype=torch.float32, device=device)
    X = deepcopy(X_real)

    for i in range(len(typeflag)):
        if typeflag[i] == 0:
            X[:, :, i] = X[:, :, i] > 0.5

    return X_real, X, GC


def generate_mixed_fMRI(p=10, typeflag=[0, 0, 1, 1, 1, 1, 1, 1, 1, 1], length_per_batch=50,
                       device=torch.device('cuda')):

    fMRI = pd.read_csv('./datasets/fmri/timeseries6.csv')
    X_np = fMRI.values

    GC_ = pd.read_csv('./datasets/fmri/GC.csv')
    GC = GC_.values
    min_max_scaler = preprocessing.MinMaxScaler()
    Y = min_max_scaler.fit_transform(X_np)
    X_pre = torch.tensor(Y[np.newaxis], dtype=torch.float32).reshape(-1, length_per_batch, p).numpy()

    # instance norm !!!
    X_real = np.zeros_like(X_pre)
    for i in range(X_real.shape[0]):
        for j in range(X_real.shape[-1]):
            instance = X_pre[i, :, j]
            instance_norm = (instance - np.min(instance)) / (np.max(instance) - np.min(instance))
            X_real[i, :, j] = instance_norm
    X_real = torch.tensor(X_real, dtype=torch.float32, device=device)
    X = deepcopy(X_real)

    for i in range(len(typeflag)):
        if typeflag[i] == 0:
            X[:, :, i] = X[:, :, i] > 0.5

    return X_real, X, GC

def generate_mixed_LotkaVolterra(p=12, typeflag=[0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1 , 1], T=1000, length_per_batch=50,
                       device=torch.device('cuda')):

    gene = MultiLotkaVolterra(p=int(p/2))
    X, GC, gc_sign = gene.simulate(t=T*10)
    X_np=X[0]

    min_max_scaler = preprocessing.MinMaxScaler()
    Y = min_max_scaler.fit_transform(X_np)
    X_pre = torch.tensor(Y[np.newaxis], dtype=torch.float32).reshape(-1, length_per_batch, p).numpy()

    # instance norm !!!
    X_real = np.zeros_like(X_pre)
    for i in range(X_real.shape[0]):
        for j in range(X_real.shape[-1]):
            instance = X_pre[i, :, j]
            instance_norm = (instance - np.min(instance)) / (np.max(instance) - np.min(instance))
            X_real[i, :, j] = instance_norm
    X_real = torch.tensor(X_real, dtype=torch.float32, device=device)
    X = deepcopy(X_real)

    for i in range(len(typeflag)):
        if typeflag[i] == 0:
            X[:, :, i] = X[:, :, i] > 0.5

    return X_real, X, GC

def generate_mixed_dream3e1(p=100, typeflag=[1 for i in range(100)], length_per_batch=50,
                       device=torch.device('cuda')):

    dream = torch.load('./datasets/dream3data/dream3E1.pt')
    X_np = dream['X'][:950]

    GC = dream['GC']

    min_max_scaler = preprocessing.MinMaxScaler()
    Y = min_max_scaler.fit_transform(X_np)
    X_pre = torch.tensor(Y[np.newaxis], dtype=torch.float32).reshape(-1, length_per_batch, p).numpy()

    # instance norm !!!
    X_real = np.zeros_like(X_pre)
    for i in range(X_real.shape[0]):
        for j in range(X_real.shape[-1]):
            instance = X_pre[i, :, j]
            instance_norm = (instance - np.min(instance)) / (np.max(instance) - np.min(instance))
            X_real[i, :, j] = instance_norm
    X_real = torch.tensor(X_real, dtype=torch.float32, device=device)
    X = deepcopy(X_real)

    for i in range(len(typeflag)):
        if typeflag[i] == 0:
            X[:, :, i] = X[:, :, i] > 0.5

    return X_real, X, GC

def generate_mixed_dream3e2(p=100, typeflag=[1 for i in range(100)], length_per_batch=50,
                       device=torch.device('cuda')):

    dream = torch.load('./datasets/dream3data/dream3E2.pt')
    X_np = dream['X'][:950]

    GC = dream['GC']

    min_max_scaler = preprocessing.MinMaxScaler()
    Y = min_max_scaler.fit_transform(X_np)
    X_pre = torch.tensor(Y[np.newaxis], dtype=torch.float32).reshape(-1, length_per_batch, p).numpy()

    # instance norm !!!
    X_real = np.zeros_like(X_pre)
    for i in range(X_real.shape[0]):
        for j in range(X_real.shape[-1]):
            instance = X_pre[i, :, j]
            instance_norm = (instance - np.min(instance)) / (np.max(instance) - np.min(instance))
            X_real[i, :, j] = instance_norm
    X_real = torch.tensor(X_real, dtype=torch.float32, device=device)
    X = deepcopy(X_real)

    for i in range(len(typeflag)):
        if typeflag[i] == 0:
            X[:, :, i] = X[:, :, i] > 0.5

    return X_real, X, GC

def generate_mixed_dream3y1(p=100, typeflag=[1 for i in range(100)], length_per_batch=50,
                       device=torch.device('cuda')):

    dream = torch.load('./datasets/dream3data/dream3Y1.pt')
    X_np = dream['X'][:950]

    GC = dream['GC']

    min_max_scaler = preprocessing.MinMaxScaler()
    Y = min_max_scaler.fit_transform(X_np)
    X_pre = torch.tensor(Y[np.newaxis], dtype=torch.float32).reshape(-1, length_per_batch, p).numpy()

    # instance norm !!!
    X_real = np.zeros_like(X_pre)
    for i in range(X_real.shape[0]):
        for j in range(X_real.shape[-1]):
            instance = X_pre[i, :, j]
            instance_norm = (instance - np.min(instance)) / (np.max(instance) - np.min(instance))
            X_real[i, :, j] = instance_norm
    X_real = torch.tensor(X_real, dtype=torch.float32, device=device)
    X = deepcopy(X_real)

    for i in range(len(typeflag)):
        if typeflag[i] == 0:
            X[:, :, i] = X[:, :, i] > 0.5

    return X_real, X, GC

def generate_mixed_dream3y2(p=100, typeflag=[1 for i in range(100)], length_per_batch=50,
                       device=torch.device('cuda')):

    dream = torch.load('./datasets/dream3data/dream3Y2.pt')
    X_np = dream['X'][:950]

    GC = dream['GC']

    min_max_scaler = preprocessing.MinMaxScaler()
    Y = min_max_scaler.fit_transform(X_np)
    X_pre = torch.tensor(Y[np.newaxis], dtype=torch.float32).reshape(-1, length_per_batch, p).numpy()

    # instance norm !!!
    X_real = np.zeros_like(X_pre)
    for i in range(X_real.shape[0]):
        for j in range(X_real.shape[-1]):
            instance = X_pre[i, :, j]
            instance_norm = (instance - np.min(instance)) / (np.max(instance) - np.min(instance))
            X_real[i, :, j] = instance_norm
    X_real = torch.tensor(X_real, dtype=torch.float32, device=device)
    X = deepcopy(X_real)

    for i in range(len(typeflag)):
        if typeflag[i] == 0:
            X[:, :, i] = X[:, :, i] > 0.5

    return X_real, X, GC

def generate_mixed_dream3y3(p=100, typeflag=[1 for i in range(100)], length_per_batch=50,
                       device=torch.device('cuda')):

    dream = torch.load('./datasets/dream3data/dream3Y3.pt')
    X_np = dream['X'][:950]

    GC = dream['GC']

    min_max_scaler = preprocessing.MinMaxScaler()
    Y = min_max_scaler.fit_transform(X_np)
    X_pre = torch.tensor(Y[np.newaxis], dtype=torch.float32).reshape(-1, length_per_batch, p).numpy()

    # instance norm !!!
    X_real = np.zeros_like(X_pre)
    for i in range(X_real.shape[0]):
        for j in range(X_real.shape[-1]):
            instance = X_pre[i, :, j]
            instance_norm = (instance - np.min(instance)) / (np.max(instance) - np.min(instance))
            X_real[i, :, j] = instance_norm
    X_real = torch.tensor(X_real, dtype=torch.float32, device=device)
    X = deepcopy(X_real)

    for i in range(len(typeflag)):
        if typeflag[i] == 0:
            X[:, :, i] = X[:, :, i] > 0.5

    return X_real, X, GC










