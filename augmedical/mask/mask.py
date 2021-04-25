from augmedical.filters.filters import BoxBlur
import torch


class UncertainMask:
    def __init__(self, channels=1, filter_size=3, iterations=1, alpha=1):
        """
        In medical imaging and segmentation tasks, the borders of the mask are more uncertain than the "solid" part.
        To reduce the impact of the exact segmentation, the label is reduced for those images.

        The idea is similar to Superpixel-Guided Label Softening for Medical Image Segmentation https://arxiv.org/pdf/2007.08897.pdf
        but we simply blur the mask with a box blur.

        this is a wrapper around boxblur...
        ----
        Args:
            channels (int, optional): [Number of channels to apply the filter to]. Defaults to 1.
            filter_size (int, optional): [description]. Defaults to 3.
            iterations (int, optional): [description]. Defaults to 1.
        """

        self.blur = BoxBlur(channels, p=1, kernel_size=filter_size, alpha=alpha, iterations=iterations)

    def __call__(self, tensor):
        if tensor.dim() == 3:
            c, w, h = tensor.shape
            return self.blur(tensor.view(1, c, w, h)).view(c, w, h)

        elif tensor.dim() == 4:
            return self.blur(tensor)
        else:
            raise Exception("not implemented", "currently only supporting 2 and 3 d images.")


class LabelSmoothing:
    # https://github.com/pytorch/pytorch/issues/7455#issuecomment-631829085
    def __init__(self, smoothing_factor=0.9) -> None:
        self.smoothing_factor = smoothing_factor

    def __call__(self, labels):
        with torch.no_grad():
            confidence = 1.0 - self.smoothing_factor

            true_dist = torch.mul(labels, confidence)
            true_dist = torch.add(true_dist, self.smoothing_factor / (labels.shape[1] - 1))

        return true_dist
