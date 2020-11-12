import numpy as np

from .ingredient import Ingredient

from .pixel import Pixel


class Threshold(Ingredient):
    def prep(self, **kwargs):
        self.lower_threshold = kwargs.get('lower_threshold', 0)
        self.upper_threshold = kwargs.pop('upper_threshold', 128)
        self.inside = kwargs.pop('inside', np.ndarray([0, 0, 0]))
        self.outside = kwargs.pop('outside', np.ndarray([255, 255, 255]))

    def cook(self, pixels: np.ndarray):
        inside_pixels = np.resize(self.inside, pixels.shape)
        outside_pixels = np.resize(self.outside, pixels.shape)

        cooked_pixels = np.full(pixels.shape, self.inside)
        binary_pixels = np.logical_or(np.average(pixels, 2) >= self.upper_threshold, np.average(pixels, 2) <= self.lower_threshold)
        cooked_pixels[binary_pixels] = self.outside[binary_pixels]

        return cooked_pixels
