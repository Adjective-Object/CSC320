from scipy.misc import imread
import numpy as np
from numpy.linalg import pinv
from p4 import debug

def matting(b1, b2, c1, c2):
    debug("calculating image matte")

    height, width = b1.shape[0:2]
    result = np.empty((height, width, 4))

    bkgcolours = np.reshape(
        np.concatenate((b1,b2), 2),
        (height, width, 6, 1))

    diffs = np.reshape(
        np.concatenate((c1-b1, c2-b2), 2),
        (height, width, 6, 1))

    # should it be repeat or tile??
    #double_identity_3 = np.repeat(np.identity(3), 2, axis = 0)
    double_identity_3 = np.tile(np.identity(3), (2,1))

    # construct As (the concatenation of 2 identities and vg_values)
    many_identities = np.reshape(double_identity_3, ((1,1,6,3)))
    many_identities = np.tile(many_identities, (height, width, 1, 1))
    
    As = np.concatenate((many_identities, bkgcolours), axis=3)
    Apinv = np.empty((height, width, 4, 6))

    # value names correspond to matrices in Part 1 of the project handout
    debug("    begin pinv & dot product..")
    for y, x in np.ndindex(b1.shape[0], b1.shape[1]):
        # debug(As[y,x], "A")
        # debug(diffs[y,x], "diffs")
        
        Apinv[y,x] = pinv(As[y,x])
        # debug(Apinv[y,x], "A+")

        result[y,x] = np.reshape(
            np.dot(Apinv[y,x], diffs[y,x]),
            (4))
    debug("    end pinv & dot product")

    #invert the alpha channel on the matte
    result[:,:,3] = -result[:,:,3]
    return np.clip(result, 0, 1);

def threshold_falloff(
        image,
        THRESHOLD=0.5,
        THRESHOLD_FALLOFF=0.5):
    debug("applying thresholding effect")


    # center around threshold and scale down, then reset offset
    falloff = (image[:,:,3] - THRESHOLD) / THRESHOLD_FALLOFF + THRESHOLD
    out = np.array(image, copy=True)

    # use the alpha value of the mask to choose between mask or threshold mask
    # all colours just for indicator of threshold mask versus mask mask
    image[:,:,3] = np.where(image[:,:,3] < THRESHOLD, image[:,:,3], falloff)

    return out;

class P1(object):
    def __init__(self, (background_a, composition_a, background_b, composition_b), mask=False):
        self.background_a = background_a
        self.background_b = background_b
        self.composition_a = composition_a
        self.composition_b = composition_b
        self.mask = mask;

    def execute(self):
        debug("performing p1 with mask =", self.mask);

        b1 = imread(self.background_a) / 255.0
        b2 = imread(self.background_b) / 255.0
        c1 = imread(self.composition_a) / 255.0
        c2 = imread(self.composition_b) / 255.0

        matted_image = matting(b1, b2, c1, c2)

        if self.mask:
            return threshold_falloff(matted_image)[:,:,3]
        else:
            return threshold_falloff(matted_image)

class P2(P1):
    def __init__(self, imgs, background):
        P1.__init__(self, imgs)
        self.background = background

    def execute(self):
        b1 = imread(self.background_a) / 255.0
        b2 = imread(self.background_b) / 255.0
        c1 = imread(self.composition_a) / 255.0
        c2 = imread(self.composition_b) / 255.0
        background = imread(self.background) / 255.0

        debug("background: ", self.background);

        foreground = threshold_falloff(matting(b1,b2,c1,c2))

        debug("matting onto background image", self.background)

        alpha = foreground[:,:,3].repeat(3).reshape(background.shape)

        return (background * (1-alpha) + foreground[:,:,0:3] * alpha)

