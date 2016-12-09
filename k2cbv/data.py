#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
:py:mod:`data.py`
-----------------

'''

from __future__ import division, print_function, absolute_import, unicode_literals
from . import CBV_DAT, CBV_SRC
import os
import logging
log = logging.getLogger(__name__)

__all__ = ['GetK2Stars']

def GetK2Stars(campaign, channels = range(99)):
  '''

  '''
  
  stars = []
  f = os.path.join(CBV_SRC, 'tables', 'c%02d.stars' % campaign)
  with open(f, 'r') as file:
    lines = file.readlines() 
    for l in lines:
      EPIC, _, channel, _ = l.split(',')
      if int(channel) in channels:
        stars.append(int(EPIC))
        
  return stars