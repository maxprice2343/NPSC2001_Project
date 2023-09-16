"""
Defines a node class that can store information about a node.
This allows an environment to define multiple nodes of different types.
"""

import numpy as np

class node:
    def __init__(self, location, range, time=0):
        self.range = range
        self.location = location
        self.time = time
        self.active = False

    def activate(self):
        self.active = True

    def deactivate(self):
        self.active = False

    def within_range(self, location):
        return np.linalg.norm(self.location, location) <= range