import random
import numpy as np
from skimage.transform import rotate
from skimage.transform import resize
import matplotlib.pyplot as plt

from ..transforms.transforms import ImageTransform


def random_float(low, high):
    return random.random() * (high - low) + low


class Stamping(ImageTransform):
    def __init__(self, path, files, p=0.1, intensity=1):
        self.aug_images = []

        for file in files:
            self.aug_images.append(plt.imread(f"{path}/{file}.png")[..., [-1]])

        self.intensity = intensity
        self.min_size = 0.5
        self.max_size = 5

        super().__init__(p=p)

    def apply(self, x, force_x=None, force_y=None):
        # get a random stamp
        stamp = self.aug_images[random.randint(0, len(self.aug_images) - 1)]

        # remember the original highest and lowest values for each channel
        min_intensities = np.min(x.reshape(-1, 3), axis=0)
        max_intensities = np.max(x.reshape(-1, 3), axis=0)

        # how much to rescale the stamp

        scale = random_float(self.min_size, self.max_size)
        stamp = resize(
            stamp, [int(stamp.shape[0] * scale), int(stamp.shape[1] * scale)]
        )

        angle = random_float(0, 360)
        stamp = rotate(stamp, angle)

        # increase size by half stamp size on all sides
        stamp_filled = np.zeros(
            [x.shape[0] + stamp.shape[0], x.shape[1] + stamp.shape[1], 1]
        )

        start_x = (
            force_x if force_x is not None else np.random.randint(0, x.shape[0])
        )
        start_y = (
            force_y if force_y is not None else np.random.randint(0, x.shape[1])
        )

        end_x = start_x + stamp.shape[0]
        end_y = start_y + stamp.shape[0]

        stamp_filled[start_x:end_x, start_y:end_y] = stamp

        # trim stamp layer
        stamp_filled = stamp_filled[
            stamp.shape[0] // 2: -stamp.shape[0] // 2,
            stamp.shape[1] // 2: -stamp.shape[1] // 2,
        ]

        if random.random() > 0.5:
            x += stamp_filled * self.intensity
        else:
            x -= stamp_filled * self.intensity

        for c in range(3):
            x[..., c] = np.clip(x[..., c], min_intensities[c], max_intensities[c])

        return x
