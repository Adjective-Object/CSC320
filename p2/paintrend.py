import os
import sys

# ##########################################################################
## Handout painting code.
###########################################################################
from PIL import Image
from pylab import *

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook

import random
import time
import math

import matplotlib.image as mpimg
import scipy as sci
from scipy.misc import imresize, imsave

from canny import *

import getopt
from debugtools import debug


from painters import *

################################
## Options and main loop logic #
################################

def print_helptext():
    debug("usage: 'paintend.py -p [version]'")
    debug("    where [version] is one of p1, 1, p2, 2 ... p6, 6")

def parse_opts():
    options, remaining_args = getopt.getopt(
        sys.argv[1:],
        'p:',
        ['part']
    )

    painter = None

    for opt, arg in options:
        if opt in ('-p', 'part'):
            if arg in (1, '1', 'p1'):
                painter = P1Painter()
            if arg in (2, '2', 'p2'):
                painter = P2Painter()
            if arg in (3, '3', 'p3'):
                painter = P3Painter()
            if arg in (4, '4', 'p4'):
                painter = P4Painter()
            if arg in (5, '5', 'p5'):
                painter = P5Painter()
            if arg in (6, '6', 'p6'):
                painter = P6Painter()
        else:
            debug("unrecognized option/argument pair '%s', '%s'" % (opt, arg))
            sys.exit(1)

    if not painter:
        print_helptext()
        sys.exit(1)

    return painter


def main():
    # select the painter based on opt
    # painter = parse_opts
    painter = parse_opts()

    # Read image and convert it to double, and scale each R,G,B
    # channel to range [0,1].
    imRGB = array(Image.open('orchid.jpg'))
    imRGB = double(imRGB) / 255.0

    plt.clf()
    plt.axis('off')

    sizeIm = imRGB.shape
    sizeIm = sizeIm[0:2]

    # alert the painter to the image
    painter.load_image(imRGB)

    # Random number seed
    np.random.seed(29645)

    time.time()
    time.clock()
    k = 0

    # paint over the image
    while painter.should_paint():
        painter.do_paint();

        k += 1
        if (k % 100 == 0):
            debug("painted stroke %s (%s remaining)" % 
                (k, len(np.where(painter.canvas == -1)[0])))

    print("done!")
    time.time()

    canvas = painter.canvas

    canvas[canvas < 0] = 0.0
    plt.clf()
    plt.axis('off')
    plt.imshow(canvas)
    plt.pause(3)
    colorImSave('output.png', canvas)


if __name__ == "__main__":
    # os.chdir("/h/u17/g2/00/g1biggse/code/csc320/a2")
    # os.chdir("/h/u17/g2/00/g4rbage/code/csc320/a2")

    np.set_printoptions(threshold=np.nan)
    main()