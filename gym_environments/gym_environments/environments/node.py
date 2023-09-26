"""
Defines a node class that can store information about a node.
This allows an environment to define multiple nodes of different types.
"""

import numpy as np

class node:
    def __init__(self, location, range, time=0):
        self.INACTIVE = 0
        self.ACTIVE = 1
        self.MISSED = -1
        self.range = range
        self.location = location
        self.time = time
        self.state = self.INACTIVE
        self.countdown = -1

    def activate(self):
        self.state = self.ACTIVE
        self.countdown = self.time
    
    def tick(self):
        if(self.countdown > 0):
            self.countdown -= 1

    def deactivate(self):
        self.state = self.INACTIVE
        #self.range = -1
        #self.countdown = -1
        #self.location = np.array([-1,-1])

    def miss(self):
        self.state = self.MISSED
        #self.range = -1
        #self.countdown = -1
        #self.location = np.array([-1,-1])

    def within_range(self, location):
        return np.linalg.norm(self.location - location) <= self.range