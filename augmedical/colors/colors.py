from skimage.color import (
    separate_stains,
    combine_stains,
    hpx_from_rgb,
    rgb_from_hpx,
    rgb2hsv,
    hsv2rgb,
)
from skimage.exposure import equalize_adapthist, rescale_intensity
import numpy as np
import random

# base class to apply to image only
from ..transforms.transforms import ImageTransform


def random_float(low, high):
    return random.random() * (high - low) + low


class AdaptiveHistogramEqualization(ImageTransform):
    def __init__(self, p=1):
        super().__init__(p=1)

    def apply(self, sample):
        return equalize_adapthist(sample["image"], clip_limit=0.03)


class Deconvolution(ImageTransform):
    def __init__(self):
        super().__init__(p=1)

    def apply(self, x, mean=[0,0,0], std=[1,1,1]):
        x = separate_stains(x, hpx_from_rgb)
        x = (x - mean)/std

        return x
    
    def fit(self, x_stack, use_median=False):
        mean_sum = [0,0,0]
        var_sum = [0,0,0]
        count = 0
        
        for x in x_stack:
            x = separate_stains(x, hpx_from_rgb)
            
            for channel in range(3):
                if use_median:
                    mean_sum[channel] += np.median(x[..., channel])
                else:
                    mean_sum[channel] += x[..., channel].mean()
                var_sum[channel] += x[..., channel].var()

                count += 1

        mean = np.array(mean_sum)/count
        std = np.sqrt(np.array(var_sum)/count)
        
        return mean, std
            
            
class Desaturation:
    def __init__(
        self,
        p=0.1,
        min_desaturation=0,
        max_desaturation=0.5,
        min_value_reduction=0.0,
        max_value_reduction=0.1,
    ):
        self.min_desaturation = min_desaturation
        self.max_desaturation = max_desaturation

        self.min_value_reduction = min_value_reduction
        self.max_value_reduction = max_value_reduction

        super().__init__(p=p)

    def apply(self, x):
        # transfere into hsv colorspace
        x = rgb2hsv(x)

        # change saturation
        x[:, :, 1] *= 1 - (
            random_float(self.min_desaturation, self.max_desaturation)
        )

        # change brightness
        x[:, :, 2] *= 1 - (
            random_float(self.min_value_reduction, self.max_value_reduction)
        )

        # dont let it go over 9000
        x = np.clip(x, 0, 1)

        # revert to good old rgb
        x = hsv2rgb(x)

        return x


class ChannelBleaching:
    def __init__(
        self,
        p=0.3,
        min_bleach=0.1,
        max_bleach=0.9,
        mode="fluorescense",
        force_channel=None,
    ):
        """
        Reduce one channels intensity, while not modifying any other channel.
        This should simulate eg a bad staining in pas while he is intact.
        """

        self.min_bleach = min_bleach
        self.max_bleach = max_bleach
        self.mode = mode
        self.force_channel = force_channel

        super().__init__(p=p)

    def apply(self, x):
        # randomly chose one channel
        selected_channel = (
            random.randint(0, x.shape[-1] - 1)
            if self.force_channel is None
            else self.force_channel
        )

        bleaching_factor = random_float(self.min_bleach, self.max_bleach)

        if self.mode == "fluorescense":
            x[..., selected_channel] = (
                x[..., selected_channel] * (1 - bleaching_factor)
                + x[..., selected_channel].min() * bleaching_factor
            )
        elif self.mode == "brightfield":
            x[..., selected_channel] = (
                x[..., selected_channel] * (1 - bleaching_factor)
                + x[..., selected_channel].max() * bleaching_factor
            )

        return x


class StainShift:
    def __init__(self, p=0.3, min_shift=1, max_shift=3, force_channel=None):
        """
        Shift one stain by n pixels in any direction.
        It is best to shift the layer that is not directly tied to the mask.

        the rolling operation might not be the best way to do it, but it is way easier and faster than manual slicing.
        """

        self.min_shift = min_shift
        self.max_shift = max_shift
        self.force_channel = force_channel

        super().__init__(p=p)

    def apply(self, x):
        # check that we dont apply filter to mask
        selected_channel = (
            random.randint(0, x.shape[-1] - 1)
            if self.force_channel is None
            else self.force_channel
        )

        rolled_dice = random.random()
        # 40% chance to do horizontal shift
        if rolled_dice <= 0.4:
            x_shift = random.randint(self.min_shift, self.max_shift)
            y_shift = 0

        # 40% chance to do vertical shift
        elif rolled_dice <= 0.8:
            x_shift = 0
            y_shift = random.randint(self.min_shift, self.max_shift)

        # 20% chance to do both shifts
        else:
            x_shift = random.randint(self.min_shift, self.max_shift)
            y_shift = random.randint(self.min_shift, self.max_shift)

        # channels, x, y
        if random.random() >= 0.5:
            x_shift *= -1

        if random.random() >= 0.5:
            y_shift *= -1

        x[..., selected_channel] = np.roll(
            x[..., selected_channel], shift=(x_shift, y_shift), axis=(0, 1)
        )

        return x
