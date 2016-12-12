import k2cbv
from k2cbv import *
from astropy.table import Table

import numpy as np
import matplotlib.pyplot as pl
import matplotlib as mpl
from k2cbv.cbv import GetChunk

mpl.style.use('seaborn-colorblind')

#To make sure we have always the same matplotlib settings
#(the ones in comments are the ipython notebook settings)

mpl.rcParams['figure.figsize']=(8.0,6.0)    #(6.0,4.0)
mpl.rcParams['font.size']=18               #10 
mpl.rcParams['savefig.dpi']= 200             #72 
mpl.rcParams['axes.labelsize'] = 16
mpl.rcParams['axes.labelsize'] = 16
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12


campaign = 4
module = 15
model = 'everest1'
max_stars = 150

# Download the light curves
time, breakpoints, fluxes = k2cbv.GetStars(campaign, module, model = model, max_stars = max_stars)

# Construct the design matrix
X = k2cbv.GetX(campaign, module, model = model, max_stars = max_stars)

fname = 'EPIC_200000004_mast.fits'

# Fit all the stars
# k2cbv.Fit_Halo(fname,campaign = campaign, module = module, model = model, 
# 	breakpoints = breakpoints, max_stars = max_stars)

lc = Table.read(fname)

halotime, flux = lc['time'], lc['flux'] - lc['trposi'] + np.nanmedian(lc['trposi'])
matching = np.array([np.where(np.abs(time-t)<0.0001)[0][0] for t in halotime])
time, fluxes, X = time[matching], fluxes[:,matching], X[matching,:]

mask = np.where(~np.isfinite(flux))
path = '.'

# Get the design matrix 
X = GetX(campaign, module, model = model)

# Loop over all the light curve segments
model = [None for b in range(len(breakpoints))]
weights = [None for b in range(len(breakpoints))]

for b in range(len(breakpoints)):

	# Get the indices for this light curve segment
	inds = GetChunk(time, breakpoints, b)
	masked_inds = GetChunk(time, breakpoints, b, mask = mask)

	# Ordinary least squares
	mX = X[masked_inds]
	A = np.dot(mX.T, mX)
	B = np.dot(mX.T, flux[masked_inds])
	weights[b] = np.linalg.solve(A, B)
	model[b] = np.dot(X[inds], weights[b])

	# Vertical alignment
	if b == 0:
		model[b] -= np.nanmedian(model[b])
	else:
		# Match the first finite model point on either side of the break
		# We could consider something more elaborate in the future
		i0 = -1 - np.argmax([np.isfinite(model[b - 1][-i]) for i in range(1, len(model[b - 1]) - 1)])
		i1 = np.argmax([np.isfinite(model[b][i]) for i in range(len(model[b]))])
		model[b] += (model[b - 1][i0] - model[b][i1])

# Join model and normalize	
model = np.concatenate(model)
model -= np.nanmedian(model)

# Plot the light curve, model, and corrected light curve
fig, ax = pl.subplots(2, figsize = (12, 6), sharex = True)
ax[0].plot(np.delete(time, mask), np.delete(flux, mask), 'k.', markersize = 3)
ax[0].plot(np.delete(time, mask), np.delete(model, mask) + np.nanmedian(flux), 'r-')
ax[1].plot(np.delete(time, mask), np.delete(flux - model, mask), 'k.', markersize = 3)
ax[0].set_ylabel('Original Flux', fontsize = 16)
ax[1].set_ylabel('Corrected Flux', fontsize = 16)
ax[1].set_xlabel('Time (BJD - 2454833)', fontsize = 16)

# Force the axis range based on the non-outliers
ax[0].margins(0.01, None)
ax[0].set_xlim(ax[0].get_xlim())
ax[0].set_ylim(ax[0].get_ylim())
ax[1].set_xlim(ax[1].get_xlim())
ax[1].set_ylim(ax[1].get_ylim())

# Now plot the outliers
ax[0].plot(time[mask], flux[mask], 'r.', markersize = 3, alpha = 0.5)
ax[1].plot(time[mask], flux[mask] - model[mask], 'r.', markersize = 3, alpha = 0.5)

pl.suptitle('Atlas Detrended' % EPIC, fontsize = 20)
fig.savefig('atlas.pdf')
pl.close()
