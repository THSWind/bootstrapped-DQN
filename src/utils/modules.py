"""
Imports all necessary modules with error handling

"""

from __future__ import print_function
import itertools
import time
import argparse
import pickle
import os
import sys
# import loky
import random
import collections
from collections import Counter, defaultdict, namedtuple
import json
import warnings as _warnings
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import logging
import hashlib


class memorize(object):
    """
    Caches a function's return value each time it is called.
    If called later with the same arguments, the cached
    value is returned.
    """
    def __init__(self, func):
        self.func = func
        self.cache = {}

    def __call__(self, *args):
        if not isinstance(args, collections.Hashable):
            return self.func(*args)
        if args in self.cache:
            return self.cache[args]
        else:
            value = self.func(*args)
            self.cache[args] = value
            return value

    def __repr__(self):
        return self.func.__doc__

    def __get__(self, obj, objtype):
        return functools.partial(self.__call__, obj)

try:
    import cv2
except ModuleNotFoundError:
    _warnings.warn('cv2 not found')

try:
    import gym
except ModuleNotFoundError:
    _warnings.warn('gym not found')

try:
    from tqdm import tqdm, trange
except ModuleNotFoundError:
    _warnings.warn('tqdm not found')


class CircularMemory:
    def __init__(self, size):
        self.size = size
        self.mem = []
        self.start_idx = 0

    def add(self, entry):
        if len(self.mem) < self.size:
            self.mem.append(entry)
        else:
            self.mem[self.start_idx] = entry
            self.start_idx = (self.start_idx + 1) % self.size

    def sample(self, n):
        return random.sample(self.mem, n)

    def __len__(self):
        return len(self.mem)

    def __getitem__(self, i):
        assert i < len(self)
        return self.mem[(self.start_idx + i) % self.size]
