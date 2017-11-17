#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 10:20:02 2017

@author: hafizimtiaz
"""
import numpy as np
from scipy.stats import ortho_group

K = 2               # number of independent components
N = 1000             # number of samples
D = 5               # dimension
tol = 1e-10
maxitr = 1000
A = ortho_group.rvs(dim = D)
A = A[:, :K]        # mixing matrix

# Data generation
mn = np.zeros([D,])
cv = np.eye(D)
h = np.random.exponential(scale = 1.0, size = [K, N])
X = np.dot(A, h) + np.sqrt(0.05) * np.random.multivariate_normal(mn, cv, N).T
np.savez('mixing_matrix', A, X)

# make distributed data
num_sites = 5
num_samples_site = np.int(N / num_sites)
st_id = 0
en_id = num_samples_site

for s in range(num_sites):
    Xs = X[:, st_id : en_id]
    hs = h[:, st_id : en_id]
    filename = "site" + str(s)
    np.save(filename, Xs)
    st_id += num_samples_site
    en_id += num_samples_site
