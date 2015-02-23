#!/local/packages/anaconda3/bin/python

# ##########################################################################
## Handout painting code.
###########################################################################

import time
import getopt

from scipy.misc import imresize

from debugtools import debug
from painters import *

################################
## Options and main loop logic #
################################



def colourImSave(filename, array):
    imArray = imresize(array, 3., 'nearest')
    if (len(imArray.shape) == 2):
        imsave(filename, cm.jet(imArray))
    else:
        imsave(filename, imArray)


def print_helptext():
    debug("usage: 'paintrend.py --part=<version> [--image=<image_name> --radius=<radius> --length=<length>]'")
    debug("    where <version> is one of p1, 1, p2, 2 ... p6, 6")
    debug("          <image_name> is the name of the image to use")
    debug("          <radius> is the radius of brush stroke")
    debug("          <length> is the maximum length of the brush stroke")

def parse_opts():
    options, remaining_args = getopt.getopt(
        sys.argv[1:],
        'pirl:',
        ['part=', 'image=', 'radius=', 'length=']
    )

    painter = None
    image_name = 'orchid.jpg'
    radius = 3
    length = 20

    for opt, arg in options:
        if opt in ('-p', '--part'):
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

        elif opt in ('-i', '--image'):
            image_name = arg

        elif opt in ('-r', '--radius'):
            radius = float(arg)

        elif opt in ('-l', '--length'):
            length = float(arg)

        else:
            debug("unrecognized option/argument pair '%s', '%s'" % (opt, arg))
            sys.exit(1)

    if not painter:
        print_helptext()
        sys.exit(1)

    painter.radius = radius
    painter.halfLen = length / 2

    return painter, image_name

def main():
    # select the painter based on opt
    # painter = parse_opts
    #painter, image_name = parse_opts()
    painter, image_name = P6Painter(), "orchid.jpg"
    painter.base_radius = 2
    painter.alpha = 0.7

    # Read image and convert it to double, and scale each R,G,B
    # channel to range [0,1].
    imRGB = array(Image.open(image_name))
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
        if k % 100 == 0:
            debug("painted stroke %s (%s pixels remaining)" % 
                (k, np.where(painter.canvas[:,:,3] < 1 )[0].size))

    print("done!")
    time.time()

    canvas = painter.canvas[:,:,0:3]

    canvas[canvas < 0] = 0.0
    plt.clf()
    plt.axis('off')
    plt.imshow(canvas)
    plt.pause(3)
    colourImSave('output.png', canvas)


if __name__ == "__main__":
    # os.chdir("/h/u17/g2/00/g1biggse/code/csc320/a2")
    # os.chdir("/h/u17/g2/00/g4rbage/code/csc320/a2")

    np.set_printoptions(threshold=np.nan)
    main()