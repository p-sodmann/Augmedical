import numpy as np


class RandAugment:
    def __init__(self, augmentations, n, m):
        self.augmentations = augmentations
        self.n = n
        self.intensity = m

    def __call__(self, sample):
        sampled_ops = np.random.choice(self.augmentations, self.n)

        for op in sampled_ops:
            sample = op(sample, self.intensity)

        return sample
