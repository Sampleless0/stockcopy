import torch
import random

class ReplayBuffer:

    def __init__(self, capacity):

        self.capacity = capacity

        # fixed size circular buffer
        self.storage = [None] * capacity

        self.position = 0
        self.size = 0

    def __len__(self):

        return self.size

    # add new training positions
    def add_samples(self, samples):

        for sample in samples:

            # keep everything on cpu
            stored_sample = {
                'board' : sample['board'].detach().to(device = 'cpu', dtype = torch.float16).contiguous(),
                'legal_ids' : sample['legal_ids'].detach().to(device = 'cpu', dtype = torch.long).contiguous(),
                'policy_target' : sample['policy_target'].detach().to(device = 'cpu', dtype = torch.float16).contiguous(),
                'value_target' : sample['value_target'].detach().to(device = 'cpu', dtype = torch.float32).contiguous()
            }

            # insert sample
            self.storage[self.position] = stored_sample

            # advance circular position
            self.position = (self.position + 1) % self.capacity

            self.size = min(self.size + 1, self.capacity)

    # randomly sample training positions
    def sample(self, num_samples):

        sample_count = min(num_samples, self.size)

        # sample without replacement
        indices = random.sample(range(self.size), sample_count)

        return [
            self.storage[i]
            for i in indices
        ]

    # save buffer state
    def state_dict(self):

        return {
            'capacity' : self.capacity,
            'storage' : self.storage,
            'position' : self.position,
            'size' : self.size
        }

    # restore buffer state
    def load_state_dict(self, state):

        old_capacity = state['capacity']
        old_storage = state['storage']
        old_position = state['position']
        old_size = state['size']

        # reconstruct samples from oldest to newest
        if old_size < old_capacity:

            ordered_samples = old_storage[:old_size]

        else:

            ordered_samples = [
                old_storage[
                    (old_position + i) % old_capacity
                ]
                for i in range(old_size)
            ]

        # reset current buffer
        self.storage = [None] * self.capacity
        self.position = 0
        self.size = 0

        # keep newest samples that fit
        self.add_samples(ordered_samples[-self.capacity:])