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


###########################################################
## Painting helper methods
###########################################################

indices_x, indices_y = None, None


def colorImSave(filename, array):
    imArray = imresize(array, 3., 'nearest')
    if (len(imArray.shape) == 2):
        imsave(filename, cm.jet(imArray))
    else:
        imsave(filename, imArray)


def markStroke(mrkd, p0, p1, rad, val):
    # Mark the pixels that will be painted by
    # a stroke from pixel p0 = (x0, y0) to pixel p1 = (x1, y1).
    # These pixels are set to val in the ny x nx double array mrkd.
    # The paintbrush is circular with radius rad>0

    sizeIm = mrkd.shape
    sizeIm = sizeIm[0:2];
    nx = sizeIm[1]
    ny = sizeIm[0]
    p0 = p0.flatten('F')
    p1 = p1.flatten('F')
    rad = max(rad, 1)
    # Bounding box
    concat = np.vstack([p0, p1])
    bb0 = np.floor(np.amin(concat, axis=0)) - rad
    bb1 = np.ceil(np.amax(concat, axis=0)) + rad
    # Check for intersection of bounding box with image.
    intersect = 1
    if ((bb0[0] > nx) or (bb0[1] > ny) or (bb1[0] < 1) or (bb1[1] < 1)):
        intersect = 0
    if intersect:
        # Crop bounding box.
        bb0 = np.amax(np.vstack([np.array([bb0[0], 1]), np.array([bb0[1], 1])]), axis=1)
        bb0 = np.amin(np.vstack([np.array([bb0[0], nx]), np.array([bb0[1], ny])]), axis=1)
        bb1 = np.amax(np.vstack([np.array([bb1[0], 1]), np.array([bb1[1], 1])]), axis=1)
        bb1 = np.amin(np.vstack([np.array([bb1[0], nx]), np.array([bb1[1], ny])]), axis=1)
        # Compute distance d(j,i) to segment in bounding box
        tmp = bb1 - bb0 + 1
        szBB = [tmp[1], tmp[0]]
        q0 = p0 - bb0 + 1
        q1 = p1 - bb0 + 1
        t = q1 - q0
        nrmt = np.linalg.norm(t)
        [x, y] = np.meshgrid(np.array([i + 1 for i in range(int(szBB[1]))]),
                             np.array([i + 1 for i in range(int(szBB[0]))]))
        d = np.zeros(szBB)
        d.fill(float("inf"))

        if nrmt == 0:
            # Use distance to point q0
            d = np.sqrt((x - q0[0]) ** 2 + (y - q0[1]) ** 2)
            idx = (d <= rad)
        else:
            # Use distance to segment q0, q1
            t = t / nrmt
            n = [t[1], -t[0]]
            tmp = t[0] * (x - q0[0]) + t[1] * (y - q0[1])
            idx = (tmp >= 0) & (tmp <= nrmt)
            if np.any(idx.flatten('F')):
                d[np.where(idx)] = abs(n[0] * (x[np.where(idx)] - q0[0]) + n[1] * (y[np.where(idx)] - q0[1]))
            idx = (tmp < 0)
            if np.any(idx.flatten('F')):
                d[np.where(idx)] = np.sqrt((x[np.where(idx)] - q0[0]) ** 2 + (y[np.where(idx)] - q0[1]) ** 2)
            idx = (tmp > nrmt)
            if np.any(idx.flatten('F')):
                d[np.where(idx)] = np.sqrt((x[np.where(idx)] - q1[0]) ** 2 + (y[np.where(idx)] - q1[1]) ** 2)

            #Pixels within crop box to paint have distance <= rad
            idx = (d <= rad)
        #Mark the pixels
        if np.any(idx.flatten('F')):
            xy = (bb0[1] - 1 + y[np.where(idx)] + sizeIm[0] * (bb0[0] + x[np.where(idx)] - 2)).astype(int)
            sz = mrkd.shape
            m = mrkd.flatten('F')
            m[xy - 1] = val
            mrkd = m.reshape(mrkd.shape[0], mrkd.shape[1], order='F')

            '''
            row = 0
            col = 0
            for i in range(len(m)):
                col = i//sz[0]
                mrkd[row][col] = m[i]
                row += 1
                if row >= sz[0]:
                    row = 0
            '''

    return mrkd


def paintStroke(canvas, x, y, p0, p1, colour, rad):
    # Paint a stroke from pixel p0 = (x0, y0) to pixel p1 = (x1, y1)
    # on the canvas (ny x nx x 3 double array).
    # The stroke has rgb values given by colour (a 3 x 1 vector, with
    # values in [0, 1].  The paintbrush is circular with radius rad>0
    sizeIm = canvas.shape
    sizeIm = sizeIm[0:2]
    idx = markStroke(np.zeros(sizeIm), p0, p1, rad, 1) > 0
    # Paint
    if np.any(idx.flatten('F')):
        canvas = np.reshape(canvas, (np.prod(sizeIm), 3), "F")
        xy = y[idx] + sizeIm[0] * (x[idx] - 1)
        canvas[xy - 1, :] = np.tile(np.transpose(colour[:]), (len(xy), 1))
        canvas = np.reshape(canvas, sizeIm + (3,), "F")
    return canvas


###########################################################
## Part 1:
##   paint random strokes while there are unfilled regions
#
###########################################################


# Orientation of paint brush strokes
DUMB_THETA = 2 * pi * np.random.rand(1, 1)[0][0]
# vector from center to one end of the stroke.
DUMB_DELTA = np.array([cos(DUMB_THETA), sin(DUMB_THETA)])

def has_unpainted_regions(canvas):
    debug("Amount left unpainted: " + str(np.where(canvas == -1)[0].size))
    return np.where(canvas == -1)[0].size > 0

def random_stroke(image, canvas, radius=3, halfLen=10):
    # Randomly select stroke center
    cntr = np.floor(
        np.random.rand(2, 1).flatten() *
        np.array([image.shape[1], image.shape[0]])) + 1
    cntr = np.amin(
        np.vstack((cntr, np.array([image.shape[1], image.shape[0]]))),
        axis=0)

    # Grab colour from image at center position of the stroke.
    colour = np.reshape(image[cntr[1] - 1, cntr[0] - 1, :], (3, 1))

    # Add the stroke to the canvas and return
    # x, ny = (image.shape[1], image.shape[0])
    length1, length2 = (halfLen, halfLen)
    return paintStroke(canvas, indices_x, indices_y,
                       cntr - DUMB_DELTA * length2,
                       cntr + DUMB_DELTA * length1,
                       colour, radius)


###########################################################
## Part 2:
##   paint random in dark areas centered on unfilled regions
##
###########################################################

def unfilled_coordinate(image):
     # get coordinates of unfilled regions
    x, y, _ = np.where(image == -1)

    # set the center to a random unfilled spot
    index = random.randint(0, len(x) - 1)

    return np.array([y[index], x[index]])

def random_stroke_on_unfilled(image, canvas, radius=3, halfLen=10):
    center = unfilled_coordinate(canvas)

    # Grab colour from image at center position of the stroke.
    colour = np.reshape(image[center[1] - 1, center[0] - 1, :], (3, 1))

    # Add the stroke to the canvas and return
    # nx, ny = (image.size[1], image.size[0])
    length1, length2 = (halfLen, halfLen)
    return paintStroke(
        canvas, indices_x, indices_y,
        center - DUMB_DELTA * length2,
        center + DUMB_DELTA * length1,
        colour, radius)

###########################################################
## Part 3:
##   Compute Canny Edgels for an image
##
###########################################################

def compute_canny_edgels(image, sigma=2.0, thresHigh=20, thresLow=4):
    red, green, blue = (image[:, :, i] for i in range(3))

    # formula taken from assignment handout
    intensity = 0.3 * red + 0.59 * green + 0.11 * blue

    # canny expects a range from 0 to 1, so normalize
    intensity = intensity / np.max(intensity)
    return canny(intensity, sigma, thresHigh=thresHigh, thresLow=thresLow)

def write_test_edgels(image):
    for i in range(1, 8):
        for j in range(1, 10):
            canny_edgels = compute_canny_edgels(image, thresHigh=i * 10,
                                                thresLow=(i * 10) / j)
            imsave("edgel_outputs/edgels_{}_{}.png".format(i, j), canny_edgels)

###########################################################
## Part 4:
##   Clipping paint strokes at Canny edges
##
###########################################################

def out_of_bounds(point, image):
    return point[0] < 0 or point[1] < 0 or \
        point[0] >= image.shape[1] or point[1] >= image.shape[0]

def walk_from(x, image, edgels, delta, max_length):
    k = 1
    s = delta / np.max(np.abs(delta))
    xk = x + np.round(k * s)
    while np.linalg.norm(xk - x) <= max_length:
        if out_of_bounds(xk, image):
            return x + np.round((k - 1) * s)
        elif edgels[xk[1], xk[0]]:
            return xk

        k += 1
        xk = x + np.round(k * s)

    return xk


def clipped_stroke(image, canvas, edgels, radius=3, halfLen=30):
    center = unfilled_coordinate(canvas)

    # Grab colour from image at center position of the stroke.
    colour = np.reshape(image[center[1] - 1, center[0] - 1, :], (3, 1))

    if edgels[center[1] - 1, center[0] - 1]:
        return paintStroke(canvas, indices_x, indices_y, center, center,
                           colour, radius)

    endpoint1 = walk_from(center, canvas, edgels, DUMB_DELTA, halfLen)
    endpoint2 = walk_from(center, canvas, edgels, -DUMB_DELTA, halfLen)

    return paintStroke(canvas, indices_x, indices_y,
                   endpoint1, endpoint2,
                   colour, radius)

###########################################################
## Part 5:
##   Orienting clipped strokes based on local derivative 
##   of image (by value)
##
###########################################################

def oriented_stroke(image, canvas, edgels, derivatives, radius=3, halfLen=10):
    center = unfilled_coordinate(canvas)

    # Grab colour from image at center position of the stroke.
    colour = np.reshape(image[center[1] - 1, center[0] - 1, :], (3, 1))

    if edgels[center[1] - 1, center[0] - 1]:
        return paintStroke(canvas, indices_x, indices_y, center, center,
                           colour, radius)
    
    #diff = derivatives[center[1], center[0]]    
    diff = np.array([derivatives[center[1], center[0], 1],
            derivatives[center[1], center[0], 0]])

    endpoint1 = walk_from(center, canvas, edgels, 
                          diff, halfLen)
    endpoint2 = walk_from(center, canvas, edgels, 
                          -diff, halfLen)
    
    return paintStroke(canvas, indices_x, indices_y,
               endpoint1, endpoint2,
               colour, radius)


###########################################################
## Part 6:
##   Adding random variations to the color, angle, radius, 
##   and intensity(jk not really?) while painting
##
###########################################################


def random_oriented_stroke(image, canvas, edgels, derivatives, 
                           radius=3, halfLen=10, 
                           colorRange=(-15.0/255, 15.0/255), 
                           angleRange=(-15.0/360*2*math.pi, 15.0/360*2*math.pi),
                           radRange=(-0.5,0.5)):
    center = unfilled_coordinate(canvas)

    # Grab colour from image at center position of the stroke.
    colour = np.reshape(image[center[1] - 1, center[0] - 1, :], (3, 1))
    colour = (  max(0, min(1, colour[0] + random.uniform(*colorRange) )),
                max(0, min(1, colour[1] + random.uniform(*colorRange) )),
                max(0, min(1, colour[2] + random.uniform(*colorRange) )))

    if edgels[center[1] - 1, center[0] - 1]:
        return paintStroke(canvas, indices_x, indices_y, center, center,
                           colour, radius)
     
    diff = derivatives[center[1], center[0] , ::-1]
    """
    theta = math.atan2(diff[0], diff[1])
    theta = theta + random.uniform(*angleRange)
    diff = np.array([sin(theta), cos(theta)])
    """

    endpoint1 = walk_from(center, canvas, edgels, 
                           diff, halfLen)
    endpoint2 = walk_from(center, canvas, edgels, 
                          -diff, halfLen)
    
    return paintStroke(canvas, indices_x, indices_y,
               endpoint1, endpoint2,
               colour, radius # + random.uniform(*radRange)
               )




################################
## Options and main loop logic #
################################

def print_helptext():
    debug("usage: 'paintend.py -p [version]'")
    debug("    where [version] is one of p1, 1, p2, 2 ... p6, 6")

def parse_opts():
    options, remaining_args = getopt.getopt(
        sys.argv[1:],
        'p',
        ['part']
    )

    should_paint, paint = None, None

    for opt, arg in options:
        if opt in ('p', 'part'):
            if arg in ('1', 'p1'):
                #TODO decide what flags to set
                should_paint = has_unpainted_regions
                paint = random_stroke
        else:
            debug("unrecognized option/argument pair %s/%s" % (opt, arg))
            sys.exit(1)

    if not (should_paint and paint):
        print_helptext()
        sys.exit(1)

    return should_paint, paint


def main():
    global indices_x, indices_y

    #should_paint, canny_clip, paint = parse_opts();
    should_paint, paint = has_unpainted_regions, random_oriented_stroke
    canny_clip = True
    should_oriented_stroke = True
    test_edgels = False

    # Read image and convert it to double, and scale each R,G,B
    # channel to range [0,1].
    imRGB = array(Image.open('orchid.jpg'))
    imRGB = double(imRGB) / 255.0

    plt.clf()
    plt.axis('off')

    sizeIm = imRGB.shape
    sizeIm = sizeIm[0:2]

    # Set up x, y coordinate images, and canvas.
    [indices_x, indices_y] = np.meshgrid(np.array([i + 1 for i in range(int(sizeIm[1]))]),
                                         np.array([i + 1 for i in range(int(sizeIm[0]))]))
    canvas = np.zeros((sizeIm[0], sizeIm[1], 3))
    canvas.fill(-1)  ## Initially mark the canvas with a value out of range.
    # Negative values will be used to denote pixels which are unpainted.

    # Random number seed
    np.random.seed(29645)

    time.time()
    time.clock()
    k = 0

    if test_edgels:
        write_test_edgels(imRGB)

    canny_edgels = None
    if canny_clip:
        canny_edgels = compute_canny_edgels(imRGB)
    
    if should_oriented_stroke:
        imIntensity = intensityImg(imRGB)
        derivatives = yx_derivatives(imIntensity)

    while should_paint(canvas) and k < 1000:
        paint_args = (imRGB, canvas, canny_edgels) if canny_clip else (imRGB, canvas)
        if should_oriented_stroke:
            paint_args = (imRGB, canvas, canny_edgels, derivatives)
        canvas = paint(*paint_args)

        k += 1
        #debug("painted stroke %s" % (k))

    print("done!")
    time.time()


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