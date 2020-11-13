import numpy as np

from .seasoning import Seasoning


class Interval(Seasoning):

    def prep(self, **kwargs):
        self.delimiter = kwargs.get('delimiter', np.array([255, 255, 255]))

    def season(self):
        pass