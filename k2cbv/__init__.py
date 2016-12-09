#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import, unicode_literals
import logging
import os, sys

# Make sure we can find K2PLR
sys.path.insert(1, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'k2plr'))

# Constants
CBV_DAT = os.path.expanduser(os.environ.get("CBV_DAT", os.path.join("~", ".k2cbv")))  
CBV_SRC = os.path.dirname(os.path.abspath(__file__))

# Set up logging
root = logging.getLogger()
root.handlers = []
root.setLevel(logging.DEBUG)
sh = logging.StreamHandler(sys.stdout)
sh.setLevel(logging.DEBUG)
sh_formatter = logging.Formatter("[%(funcName)s()]: %(message)s")
sh.setFormatter(sh_formatter)
root.addHandler(sh)

# Imports
from .data import *
from .cbv import *