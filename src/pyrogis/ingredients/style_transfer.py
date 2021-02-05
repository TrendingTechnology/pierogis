import tensorflow as tf
import numpy as np

from .ingredient import Ingredient
from .pierogi import Pierogi


class StyleTransfer(Ingredient):
    def prep(self, style: Pierogi):
        self.style = style

    def cook(self, pixels: np.ndarray):
        style_array = self.style.pixels

        cooked_pixels = tf.style_transfer(style_array, pixels)

        return np.array(cooked_pixels * 255).astype(np.dtype('uint8'))
