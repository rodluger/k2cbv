#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
:py:mod:`cbv.py`
----------------

'''

from __future__ import division, print_function, absolute_import, unicode_literals
from . import CBV_DAT, CBV_SRC
from .data import GetK2Stars
import k2plr
import numpy as np
from scipy.signal import savgol_filter, medfilt
import george
import matplotlib.pyplot as pl
from matplotlib.ticker import MaxNLocator
import os, sys
from urllib2 import HTTPError
import logging
from astropy.table import Table

log = logging.getLogger(__name__)

__all__ = ['FitAll', 'Fit_Halo','Fit', 'GetX', 'GetStars']

def GetLightCurve(EPIC, campaign, model = 'everest1', **kwargs):
  '''
  
  '''
  
  # De-trended light curve file name
  if model == 'everest1':
    data = k2plr.EVEREST(EPIC, version = 1)
    time = data.time
    flux = data.flux
    mask = data.mask
    breakpoints = data.breakpoints
  elif model == 'k2sc':
    data = k2plr.K2SC(EPIC)
    time = data.time
    flux = data.pdcflux
    mask = data.mask
    breakpoints = data.breakpoints
  else:
    raise NotImplementedError("")
    
  return time, flux, breakpoints, mask
  
def GetStars(campaign, module, max_stars = None, model = 'everest1', **kwargs):
  '''
  
  '''
  
  # Get the channel numbers
  channels = Channels(module)
  
  # Get the EPIC numbers
  stars = GetK2Stars(campaign, channels)[:max_stars]
  N = len(stars)
  assert N > 0, "No light curves found."

  # Loop over all stars and store the fluxes in a list
  fluxes = []
  masks = []
  for n in range(N):
    
    log.info("Downloading light curve %d/%d..." % (n + 1, N))
    
    # Get the data
    try:
      if model == 'everest1':
        data = k2plr.EVEREST(stars[n], version = 1) 
        t = data.time
        y = data.flux
        m = data.mask
        breakpoints = data.breakpoints
      elif model == 'k2sc':
        data = k2plr.K2SC(stars[n])
        t = data.time
        y = data.pdcflux
        m = data.mask
        breakpoints = data.breakpoints
      else:
        raise NotImplementedError("")
    except HTTPError:
      # Target not found
      continue
      
    if n == 0:
      time = t
    
    # Interpolate over outliers 
    y = np.interp(t, np.delete(t, m), np.delete(y, m))
    
    # Throw out light curves with more than 20% variation 
    # above/below the median, as they could mess up PCA
    my = np.nanmedian(y)
    if len(np.where((y < 0.8 * my) | (y > 1.2 * my))[0]):
      continue
    
    # Append fluxes to our running list
    fluxes.append(y)
    
    # Append the outlier mask to our running list
    mask = np.zeros_like(y)
    mask[m] = 1
    masks.append(mask)
  
  # Make data points that are missing in all light curves NaNs
  masks = np.array(masks)
  bad = np.where(np.sum(masks, axis = 0) == masks.shape[0])[0]
  fluxes = np.array(fluxes)
  fluxes[:,bad] = np.nan
  
  return time, breakpoints, fluxes

def SavGol(t, y, svwin = 99, svorder = 2, **kwargs):
  '''
  Masks NaNs and applies a Savitsky-Golay low-pass filter to `y`
  We might have to play with the parameters. A window size of 99
  cadences seems to work well for Everest, but I think we need a 
  larger window (or a different filter) for K2SC.
  
  '''
  
  # Linearly interpolate over nans
  nans = np.where(np.isnan(y))
  t_ = np.delete(t, nans)
  y_ = np.delete(y, nans, axis = 0)
  yfin = np.array(y)
  yfin[nans] = np.interp(t[nans], t_, y_)
  
  # Apply the savgol filter
  yfin = savgol_filter(yfin, svwin, svorder)
  
  # Re-populate the nans
  yfin[nans] = np.nan
  
  return yfin

def GetChunk(time, breakpoints, b, mask = []):
  '''
  Returns the indices corresponding to a given light curve chunk.
  
  :param int b: The index of the chunk to return

  '''

  M = np.delete(np.arange(len(time)), mask, axis = 0)
  if b > 0:
    res = M[(M > breakpoints[b - 1]) & (M <= breakpoints[b])]
  else:
    res = M[M <= breakpoints[b]]
  return res

def fObj(res, **kwargs):
  '''
  SOM likelihood function. Nothing fancy.
  
  '''
  
  return -0.5 * np.nansum(res ** 2)

def Channels(module):
  '''
  Returns the channels contained in the given K2 module.
  
  '''
  
  nums = {2:1, 3:5, 4:9, 6:13, 7:17, 8:21, 9:25, 
          10:29, 11:33, 12:37, 13:41, 14:45, 15:49, 
          16:53, 17:57, 18:61, 19:65, 20:69, 22:73, 
          23:77, 24:81}
  
  if module in nums:
    return [nums[module], nums[module] + 1, 
            nums[module] + 2, nums[module] + 3]
  else:
    return None

def SOM(time, fluxes, nflx, alpha_0 = 0.1, ki = 3, kj = 1, 
        niter = 30, periodic = False, **kwargs):
  '''
  Computes a self-organizing map and returns the arrays
  in each of the pixels/neurons.
  
  :param array_like time: The time array
  :param array_like fluxes: A list of the flux arrays for all the targets
  :param array_like nflx: The normalized flux arrays
  :param float alpha_0: The amplitude of the pixel update step. Default 0.1
  :param int ki: The number of horizontal cells. Default 3
  :param int kj: The number of vertical cells. Defaul 1
  :param int niter: The number of SOM iterationrs. Default 30
  :param bool periodic: Periodic grid boundary conditions? Default `False`
  
  ''' 
   
  # Initialize the SOM
  N = len(fluxes)
  t = 0
  sig_0 = max(ki, kj)
  alpha = alpha_0
  sig = sig_0
  som = np.random.random((ki, kj, len(time))) 
    
  # Normalize the fluxes
  if nflx is None:
    maxf = np.nanmax(fluxes, axis = 0)
    minf = np.nanmin(fluxes, axis = 0)
    nflx = (fluxes - minf) / (maxf - minf)

  # Do `niter` iterations of the SOM
  for t in range(niter):
  
    # Loop over all light curves
    for n in range(N):
  
      # Find best matching pixel for this light curve
      bi, bj = np.unravel_index(np.nanargmax([fObj(nflx[n] - som[i][j]) 
                                           for i in range(ki) for j in range(kj)]), 
                                           (ki, kj))
    
      # Trick: flip light curve upside-down and see if we get a better match
      bi_, bj_ = np.unravel_index(np.nanargmax([fObj(1 - nflx[n] - som[i][j]) 
                                             for i in range(ki) for j in range(kj)]), 
                                             (ki, kj))
      if fObj(1 - nflx[n] - som[bi_][bj_]) > fObj(nflx[n] - som[bi][bj]):
        nflx[n] = 1 - nflx[n]
        bi, bj = bi_, bj_
    
      # Update the Kohonen layer
      for i in range(ki):
        for j in range(kj):
        
          # Compute distance
          if periodic:
            dx2 = min((bi - i) ** 2, (ki - (bi - i)) ** 2)
            dy2 = min((bj - j) ** 2, (kj - (bj - j)) ** 2)
          else:
            dx2 = (bi - i) ** 2
            dy2 = (bj - j) ** 2
          d2 = dx2 + dy2
        
          # Update the pixel
          som[i][j] += alpha * np.exp(-d2 / (2 * sig ** 2)) * (nflx[n] - som[i][j])
  
    # Update the params
    sig = sig_0 * np.exp(-t / niter * np.log(max(ki, kj)))
    alpha = alpha_0 * (1 - t / niter)
  
  return som

def GetRegressor(fluxes, nflx, som, **kwargs):
  '''
  Returns the array in the SOM pixel with the largest number of
  light curves. This will be used as one of our regressors
  
  :param array_like fluxes: The list of fluxes
  :param array_like nflx: The list of normalized fluxes
  :param array_like som: The SOM array
  
  '''
  
  # Find the best matching neuron for each light curve.
  # Store the indices in the `lcinds` map.
  ki, kj, sz = som.shape
  N = len(fluxes)
  lcinds = [[[] for j in range(kj)] for i in range(ki)]
  evratio = [[[] for j in range(kj)] for i in range(ki)]
  for n in range(N):
    bi, bj = np.unravel_index(np.argmin([np.nansum((nflx[n] - som[i][j]) ** 2) 
                                         for i in range(ki) for j in range(kj)]), 
                                         (ki, kj))
    lcinds[bi][bj].append(n)
  
  # We're going to use the SOM pixel with the largest number
  # of light curves as our CBV regressor
  lflat = [lcinds[i][j] for i in range(ki) for j in range(kj)]
  sflat = np.array([som[i][j] for i in range(ki) for j in range(kj)])
  pixel = np.nanargmax([len(n) for n in lflat])
  ij = np.unravel_index(pixel, (ki, kj))
    
  # Plot the results
  PlotSOM(nflx, som, lcinds, ij, **kwargs)
  
  return sflat[pixel]

def PlotSOM(nflx, som, lcinds, ij, figname = 'som.pdf', **kwargs):
  '''
  Plots the SOM.
  
  '''
  
  # Setup
  ki, kj, _ = som.shape
  N, sz = nflx.shape
  fig, ax = pl.subplots(1, figsize = (ki * 5, kj * 5))
  fig.subplots_adjust(left = 0.01, right = 0.99, bottom = 0.02, top = 0.98)
  
  # Plot the light curves, and SOMs / PCs in each pixel
  for i in range(ki):
    for j in range(kj):
      
      # The light curves
      for n in lcinds[i][j]:
        ax.plot(np.linspace(i + 0.5 - 0.4, i + 0.5 + 0.4, sz), 
                j + 0.5 - 0.4 + 0.8 * nflx[n], 
                'k-', alpha = min(1., 100. / N))
      
      # The Kohonen pixel
      ax.plot(np.linspace(i + 0.5 - 0.4, i + 0.5 + 0.4, sz), 
              j + 0.5 - 0.4 + 0.8 * som[i][j], 'r-', lw = 2)
      
      # The mean light curve, just for reference
      ax.plot(np.linspace(i + 0.5 - 0.4, i + 0.5 + 0.4, sz), 
              j + 0.5 - 0.4 + 0.8 * np.mean([nflx[n] for n in lcinds[i][j]], axis = 0), 'b-')
      
      # Indicate the number of light curves in this pixel
      ax.annotate('%d' % (len(lcinds[i][j])), 
                  xy = (i + 0.5, j + 0.95), 
                  xycoords = 'data', va = 'center', ha = 'center', 
                  zorder = 99, fontsize = 18, color = 'b' if ij == (i,j) else 'k')
      
  ax.get_xaxis().set_major_locator(MaxNLocator(integer=True))
  ax.get_yaxis().set_major_locator(MaxNLocator(integer=True))
  ax.grid(True, which = 'major', axis = 'both', linestyle = '-')
  ax.set_xlim(0, ki)
  ax.set_ylim(0, kj)
  ax.set_xticklabels([])
  ax.set_yticklabels([])
  fig.savefig(figname)
  pl.close()

def Compute(time, fluxes, breakpoints, smooth_in = True, smooth_out = True, 
            nreg = 2, path = '', **kwargs):
  '''
  Compute the CBV design matrix based on a self-organizing map.
  
  :param array_like time: The time array
  :param array_like fluxes: The list of fluxes
  :param array_like breakpoints: The indices at which the light curves are to be split. \
                                 The last element of `breakpoints` should be the index of 
                                 the last data point in the timeseries.
  :param bool smooth_in: Smooth the fluxes prior to computing the SOM? Default `True`
  :param bool smooth_out: Smooth the SOM arrays prior to fitting? Default `True`
  :param int nreg: The number of regressors; this is equal to the number of SOM recursions. \
                   Default 2
  :param str path: The path to the output directory. Default ""
  
  '''
    
  X = [None for b in range(len(breakpoints))]
  for b in range(len(breakpoints)):
    
    # Get the indices for this light curve segment
    inds = GetChunk(time, breakpoints, b)
    t = time[inds]
    f = np.array(fluxes[:, inds])
    
    # Smooth the fluxes to compute the SOM?
    if smooth_in:
      for i in range(len(f)):
        f[i] = SavGol(t, f[i], **kwargs)
    
    # The design matrix for the current chunk
    X[b] = np.ones_like(t).reshape(-1, 1)
    
    # Run `nreg` iterations
    for n in range(nreg):
      
      log.info('Computing: segment %d/%d, iteration %d/%d...' %
              (b + 1, len(breakpoints), n + 1, nreg))
      
      # Normalize the fluxes
      maxf = np.nanmax(f, axis = 1).reshape(-1, 1)
      minf = np.nanmin(f, axis = 1).reshape(-1, 1)
      nflx = (f - minf) / (maxf - minf)

      # Compute the SOM
      som = SOM(t, f, nflx, **kwargs)
    
      # Now get the regressor
      cbv = GetRegressor(f, nflx, som, figname = 
                         os.path.join(path, 'Seg%02d_Iter%02d.pdf' % 
                         (b + 1, n + 1)), **kwargs)
      if smooth_out:
        cbv = SavGol(t, cbv, **kwargs)
      
      # Append to our design matrix
      X[b] = np.hstack([X[b], cbv.reshape(-1, 1)])
      
      # Recursion. We de-trend each of the light curves with
      # our design matrix so far, then repeat. The tricky bit
      # is masking the NaNs so we don't get linalg errors.
      if n < nreg - 1:

        nans = np.where(np.isnan(X[b][:,1]))[0]
        fins = np.delete(np.arange(len(t)), nans)
        mX = np.delete(X[b], nans, axis = 0)
        A = np.dot(mX.T, mX)
        for i in range(len(f)):
          mf = np.delete(fluxes[i,inds], nans)
          f[i][fins] = mf - np.dot(mX, np.linalg.solve(A, np.dot(mX.T, mf)))
          if smooth_in:
            f[i] = SavGol(t, f[i], **kwargs)

  return X

def GetX(campaign, module, model = 'everest1', clobber = False, **kwargs):
  '''
  Get the design matrix for a given K2 campaign and module.
  
  '''
  
  # Output path
  path = os.path.join(CBV_DAT, 'c%02d' % campaign, '%2d' % module, model)
  if not os.path.exists(path):
    os.makedirs(path)
   
  # Get the design matrix
  xfile = os.path.join(path, 'X.npz')
  if clobber or not os.path.exists(xfile):
    
    # Get the light curves
    log.info('Obtaining light curves...')
    lcfile = os.path.join(path, 'lcs.npz')
    if clobber or not os.path.exists(lcfile):
      time, breakpoints, fluxes = GetStars(campaign, module, model = model, **kwargs)
      np.savez(lcfile, time = time, breakpoints = breakpoints, fluxes = fluxes)
    else:
      lcs = np.load(lcfile)
      time = lcs['time']
      breakpoints = lcs['breakpoints']
      fluxes = lcs['fluxes'][:kwargs.get('max_stars', None)]
    
    # Get the design matrix  
    X = Compute(time, fluxes, breakpoints, path = path, **kwargs)
    X = np.vstack(X)
    np.savez(xfile, X = X)
  
  else:
      
    X = np.load(xfile)['X']
  
  return X

def Fit(EPIC, campaign = None, module = None, model = 'everest1', **kwargs):
  '''
  Fit a given K2 target with the SOM design matrix and plot the results.
  
  '''
  
  # Get the data
  if campaign is None:
    campaign = Campaign(EPIC)
  if module is None:
    module = Module(EPIC)
  time, flux, breakpoints, mask = GetLightCurve(EPIC, campaign, model = model, **kwargs)
  path = os.path.join(CBV_DAT, 'c%02d' % campaign, '%2d' % module, model)
  
  # Get the design matrix  
  X = GetX(campaign, module, model = model, **kwargs)
  
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
  
  pl.suptitle('EPIC %d' % EPIC, fontsize = 20)
  fig.savefig(os.path.join(path, '%d.pdf' % EPIC))
  pl.close()

def Fit_Halo(fname, campaign = None, module = None, breakpoints = [],
  model = 'everest1', **kwargs):
  '''
  Fit a given K2 target with the SOM design matrix and plot the results.
  
  '''
  
  # Get the data
  lc = Table.read(fname)

  time, flux = lc['time'], lc['flux'] - lc['trposi'] + np.nanmedian(lc['trposi'])
  mask = np.where(np.isfinite(flux))
  path = '.'
  
  # Get the design matrix  
  X = GetX(campaign, module, model = model, **kwargs)
  
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
  
  print model

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

  return 
  
def FitAll(campaign, module, model = 'everest1', max_stars = None, **kwargs):
  '''
  Fit all the light curves in the given campaign and module.
  
  '''

  channels = Channels(module)
  stars = GetK2Stars(campaign, channels)[:max_stars]
  N = len(stars)
  for n in range(N):
    log.info("Processing light curve %d/%d..." % (n + 1, N))
    try:
      Fit(stars[n], campaign = campaign, module = module, max_stars = max_stars, model = model, **kwargs)
    except HTTPError:
      continue