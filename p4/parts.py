from scipy.misc import imread
import numpy as np
from numpy.linalg import pinv

EPSILON = 0.0001

def matting(b1, b2, c1, c2):

    # value names correspond to matrices in Part 1 of the project handout
    result = np.empty((b1.shape[0], b1.shape[1], 4))

    for y, x in np.ndindex(b1.shape[0], b1.shape[1]):

        def rgb(array):
            return array[y, x, :]

        br1, bg1, bb1 = rgb(b1)
        br2, bg2, bb2 = rgb(b2)

        # I would ideally want to use np.concatenate here to combine b1 and b2 
        # into a single column vector, but it creates an array of values as 
        # opposed to a vector, and converting it didn't seem easy
        bg_values = np.array([[br1], [bg1], [bb1], [br2], [bg2], [bb2]])

        A = np.concatenate([np.repeat(np.identity(3), 2, axis = 0), bg_values], axis = 1)
        Aplus = pinv(A)
        b = np.concatenate([rgb(c1) - rgb(b1), rgb(c2) - rgb(b2)])

        # TODO: post-processing
        result[y, x, :] = Aplus.dot(b)

    return np.where(np.clip(result, 0, 1) > EPSILON, 0, 1)

class P1(object):
    def __init__(self, (background_a, composition_a, background_b, composition_b)):
        self.background_a = background_a
        self.background_b = background_b
        self.composition_a = composition_a
        self.composition_b = composition_b

    def execute(self):
        b1 = imread(self.background_a) / 255.0
        b2 = imread(self.background_b) / 255.0
        c1 = imread(self.composition_a) / 255.0
        c2 = imread(self.composition_b) / 255.0

        matted_image = matting(b1, b2, c1, c2)
        return matted_image

class P2(P1):
    def __init__(self, imgs, background):
        P1.init(self, imgs)
        self.background = background

    def execute(self):
        pass
