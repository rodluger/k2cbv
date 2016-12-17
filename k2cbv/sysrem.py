#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
sysrem.py
---------

'''

import numpy as np
np.random.seed(1234)
import matplotlib.pyplot as pl
from sklearn.decomposition import PCA
import george

# User
tlen = 1000     # Number of cadences
nsys = 2        # Number of systematic components
nrec = 2        # Number of components to recover
nflx = 500      # Number of light curves
ncol = 3        # Number of columns to plot
nrow = 3        # Number of rows to plot
awht = 1        # Amplitude of white noise term
astr = 5.       # Amplitude of stellar variability
tstr = 0.5      # Timescale of stellar variability
asys = 1.       # Amplitude of instrumental variability
tsys = 0.1      # Timescale of instrumental variability
niter = 5       # Number of SysRem iterations
usegp = True    # Do SysRem + GP?

# Set up the plot
fig, axes = pl.subplots(nrow, ncol, sharex = True, sharey = True, figsize = (12, 8))
fig.subplots_adjust(left = 0.05, right = 0.95, top = 0.95, bottom = 0.05, wspace = 0.025, hspace = 0.025)
for i, ax in enumerate(axes.flatten()):
  ax.set_xticklabels([])
  ax.set_yticklabels([])

# The time array
time = np.linspace(0, 1, tlen)

# The intrinsic fluxes (stellar variability + a little white noise)
gp_stellar = george.GP(astr ** 2 * george.kernels.Matern32Kernel(tstr ** 2))
gp_stellar_gen = george.GP(astr ** 2 * george.kernels.Matern32Kernel(tstr ** 2))

truths = np.zeros((nflx,tlen))
tstrs = (np.random.randn(nflx)/10. + tstr)**2.

for j in range(nflx):
  gp_stellar_gen.kernel[1] = np.log10(tstrs[j])
  gp_stellar_gen.compute(time)
  truths[j,:] = gp_stellar_gen.sample()
truths += awht * np.random.randn(nflx, tlen)

# The systematic components
gp_sys = george.GP(asys ** 2 * george.kernels.Matern32Kernel(tsys ** 2))
systematics = gp_sys.sample(time, size = nsys)
fluxes = np.zeros_like(truths)
for i in range(nflx):
  fluxes[i] = truths[i] + np.dot(np.random.randn(nsys), systematics)

# Subtract the median
fluxes -= np.nanmedian(fluxes, axis = 1).reshape(-1,1)

# Assign constant errors (for now)
errors = awht * np.ones_like(fluxes)
invvar = 1. / errors ** 2

# Plot the input fluxes
for i, ax in enumerate(axes.flatten()):
  ax.plot(time, fluxes[i], 'b-', alpha = 0.5)

# SysRem
residuals = np.array(fluxes)
models = np.zeros((nflx, tlen))

# Recover `nrec` components
for n in range(nrec):
  
  c = np.zeros(nflx)
  a = np.ones(tlen)
  
  # Perform `niter` iterations
  ri = residuals * invvar
  for i in range(niter):
    
    #
    print("Running iteration %d/%d..." % (i + 1, niter))
    
    # Compute the `c` vector (the weights)
    if not usegp:
      # This is the solution for a diagonal covariance matrix for each light curve
      c = np.dot(ri, a) / np.dot(invvar, a ** 2)
    
    else:
      # This is the solution allowing red noise in each light curve.
      # Takes a lot longer, but seems to work really well.
      X = a.reshape(-1,1)
      for j in range(nflx):
        # This next step is a time sink. Perhaps we could try the HODLR solver
        gp_stellar.compute(time, errors[j])
        A = np.dot(X.T, gp_stellar.solver.apply_inverse(X))
        B = np.dot(X.T, gp_stellar.solver.apply_inverse(residuals[j]))
        c[j] = np.linalg.solve(A, B)
    
    # Compute the `a` vector (the regressors)
    a = np.dot(c, ri) / np.dot(c ** 2, invvar)
  
  # The linear model for this step
  model = np.outer(c, a)
  
  # Add to running model
  models += model
  
  # Remove this component
  residuals = residuals - model

# Plot results
for i, ax in enumerate(axes.flatten()):
  ax.plot(time, models[i] - np.nanmedian(models[i]), 'r-')
  ax.plot(time, fluxes[i] - truths[i] + np.nanmedian(truths[i]), color = 'purple')

# Do PCA for comparison
pca = PCA(n_components = nsys)
X = pca.fit_transform(fluxes.T)

# Fit and plot results
for i, ax in enumerate(axes.flatten()):
  
  if not usegp:
    # White noise only
    A = np.dot(X.T, X)
    B = np.dot(X.T, fluxes[i])
  else:
    # Red noise
    gp_stellar.compute(time, errors[i+10])
    A = np.dot(X.T, gp_stellar.solver.apply_inverse(X))
    B = np.dot(X.T, gp_stellar.solver.apply_inverse(fluxes[i+10]))
  
  w = np.linalg.solve(A, B)
  model = np.dot(X, w)
  ax.plot(time, model - np.nanmedian(model), color = 'orange')

pl.show()