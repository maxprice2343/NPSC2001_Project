import random
import numpy as np
from collections import deque

class ReplayBuffer:
    def __init__(self):
        self.gameplay_experiences = deque(maxlen=1000000)


    def store_gameplay_experience(self, state, next_state, reward, action, done):
        """
        Stores a gameplay experience in the buffer
        """
        self.gameplay_experiences.append((state, next_state, reward, action, done))
    
    def sample_gameplay_batch(self):
        """
        Samples a batch of gameplay experiences
        """
        # The experience batch size is 128, or the number of experiences,
        # whichever is smaller
        batch_size = min(128, len(self.gameplay_experiences))
        sampled_gameplay_batch = random.sample(self.gameplay_experiences, batch_size)
        state_batch, next_state_batch, action_batch, reward_batch, done_batch = [], [], [], [], []
        # Iterates through the sampled gameplay experiences and copies the
        for gameplay_experience in sampled_gameplay_batch:
            state_batch.append(gameplay_experience[0])
            next_state_batch.append(gameplay_experience[1])
            reward_batch.append(gameplay_experience[2])
            action_batch.append(gameplay_experience[3])
            done_batch.append(gameplay_experience[4])

        return np.array(state_batch), np.array(next_state_batch), action_batch, reward_batch, done_batch