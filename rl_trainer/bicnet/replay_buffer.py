import random
import numpy as np


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.replay_buffer = []
        self.max_size = buffer_size
        self.batch_size = batch_size

    def push(self, state, logits, reward, next_state, done):
        transition_tuple = (state, logits, reward, next_state, done)
        if len(self.replay_buffer) >= self.max_size:
            self.replay_buffer.pop(0)
        self.replay_buffer.append(transition_tuple)

    def get_batches(self):
        sample_batch = random.sample(self.replay_buffer, self.batch_size)

        state_batches = np.array([_[0] for _ in sample_batch])
        action_batches = np.array([_[1] for _ in sample_batch])
        reward_batches = np.array([_[2] for _ in sample_batch])
        next_state_batches = np.array([_[3] for _ in sample_batch])
        done_batches = np.array([_[4] for _ in sample_batch])

        return state_batches, action_batches, reward_batches, next_state_batches, done_batches

    def __len__(self):
        return len(self.replay_buffer)
