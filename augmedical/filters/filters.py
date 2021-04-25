from skimage.filters import gaussian
import numpy as np
import random
from ..transforms.transforms import ImageTransform


class Blur(ImageTransform):
    def __init__(self, p=0, kernel_size=1, alpha=0.5, iterations=1):
        self.kernel_size = kernel_size
        self.alpha = alpha
        self.iterations = iterations
        super().__init__(p=p)

    def filter(self, x):
        return x

    def apply(self, x) -> np.ndarray:
        if random.random() <= self.p:
            for _ in range(self.iterations):
                x = self.filter(x) * self.alpha + x * (1-self.alpha)

        return x


class GaussianBlur(Blur):
    def filter(self, x):
        return gaussian(x, sigma=self.kernel_size/4)


class BoxBlur(Blur):
    # TODO Implement
    def filter(self, x):
        return x