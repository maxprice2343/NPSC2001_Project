import numpy as np
from collections import deque
from random import sample

class transition:
    """
    Stores information about an environment state transition:
    Current state, action, reward, next state, and terminated (if the
    'next state' is terminal)
    """
    def __init__(self, state, action, reward, next_state, terminated):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.terminated = terminated

class replay_buffer:
    """
    A queue of transition objects that are used for network training
    """
    def __init__(self, buffer_size, batch_size):
        self.batch_size = batch_size

        self.buffer = deque(maxlen=buffer_size)

    def sample(self):
        """
        Returns a random sample from the buffer of length batch_size
        """
        s = None
        # The buffer needs to have at least the number of elements in a batch
        if len(self.buffer) >= self.batch_size:
            s = sample(self.buffer, self.batch_size)
        
        return s

    def length(self):
        return len(self.buffer)