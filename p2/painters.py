# -*- coding: utf-8 -*-
"""
Created on Sun Feb 15 14:21:03 2015

@author: g4rbage
"""

from pylab import *
from scipy.misc import imsave

from canny import *
import time

class Painter(object):
    image = None
    canvas = None
    radius = 3
    halfLen = 5
    alpha = 1

    def load_image(self, image):
        self.image = image
        sizeIm = image.shape
        
        self.canvas = np.zeros((sizeIm[0], sizeIm[1], 4))

    def should_paint(self):
        raise NotImplementedError
    
    def get_paint_coord(self):
        raise NotImplementedError

    def get_colour(self, point):
        raise NotImplementedError

    def get_stroke_endpoints(self, point):
        raise NotImplementedError

    def do_paint(self):
        # pick a coordinate to paint
        center = self.get_paint_coord()
        colour = self.get_colour(center)
        endpoint1, endpoint2 = self.get_stroke_endpoints(center)

        self.canvas =  paintStroke(self.canvas,
                        endpoint1, endpoint2,
                        colour, self.radius,
                        self.alpha)

###########################################################
## Part 1:
##   paint random strokes while there are unfilled regions
#
###########################################################

class P1Painter(Painter):

    def __init__(self):
        Painter.__init__(self);
        # Orientation of paint brush strokes
        # vector from center to one end of the stroke.
        theta = 2 * pi * np.random.rand(1, 1)[0][0]
        self.DUMB_DELTA = np.array([cos(theta), sin(theta)])


    def should_paint(self):
        return np.where(self.canvas[:,:,3] < 1 )[0].size > 0
    
    def get_paint_coord(self):
        cntr = np.floor(
            np.random.rand(2, 1).flatten() *
            np.array([self.image.shape[1], self.image.shape[0]])) + 1
        cntr = np.amin(
            np.vstack((cntr, np.array([self.image.shape[1], self.image.shape[0]]))),
            axis=0)
        return cntr

    def get_colour(self, cntr):
        return np.reshape(self.image[cntr[1] - 1, cntr[0] - 1, :], (3, 1))

    def get_stroke_endpoints(self, cntr):
        return (cntr + self.DUMB_DELTA * self.halfLen,
                cntr - self.DUMB_DELTA * self.halfLen)

###########################################################
## Part 2:
##   paint random in dark areas centered on unfilled regions
##
###########################################################


class P2Painter(P1Painter):

    def get_paint_coord(self):
        # get coordinates of unfilled regions
        # print(np.where(self.canvas[:,:,3] < 1))
        x, y = np.where(self.canvas[:,:,3] < 1)

        # set the center to a random unfilled spot
        
        index = random.randint(0, len(x) - 1) if len(x) > 1 else 0

        return np.array([y[index] + 1, x[index] + 1])

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


class P3Painter(Painter):

    def __init__(self):
        Painter.__init__(self);

    def load_image(self, image):
        Painter.load_image(self, image)

        # compute the edgel of loaded image and store into the canvas on load
        self.canvas = compute_canny_edgels(self.image)

    # circumvent the paint loop, instead just computing the canny edgels on load
    def should_paint(self):
        return False

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

class P4Painter(P2Painter):

    edgels = None

    def load_image(self, image):
        Painter.load_image(self, image)

        # compute the edgel of loaded image and store into the canvas on load
        self.edgels = compute_canny_edgels(self.image)

    def get_stroke_endpoints(self, center):
        return ( walk_from(
                    center, self.canvas, self.edgels, 
                    self.DUMB_DELTA, self.halfLen)
               , walk_from(
                    center, self.canvas, self.edgels, 
                    -self.DUMB_DELTA, self.halfLen) )


###########################################################
## Part 5:
##   Orienting clipped strokes based on local derivative 
##   of image (by value)
##
###########################################################

class P5Painter(P4Painter):

    intensity_gradient = None

    def load_image(self, image):
        P4Painter.load_image(self, image)
        imIntensity = intensityImg(image)
        self.intensity_gradient = yx_derivatives(imIntensity)

    def get_stroke_endpoints(self, center):

        gradY = center[1] - 1 if (center[1] >= self.intensity_gradient.shape[0]) else center[1]
        gradX = center[0] - 1 if (center[0] >= self.intensity_gradient.shape[1]) else center[0]
        diff = np.array(
            [self.intensity_gradient[gradY, gradX, 1],
             self.intensity_gradient[gradY, gradX, 0]])
 
        return ( walk_from(
                    center, self.canvas, self.edgels, 
                    diff, self.halfLen)
               , walk_from(
                    center, self.canvas, self.edgels, 
                    -diff, self.halfLen) )


###########################################################
## Part 6:
##   Adding random variations to the colour, angle, radius, 
##   and intensity(jk not really?) while painting
##
###########################################################

# from https://librat.wikispaces.com/file/view/utilities.py
def rotate2D(xt, yt, angle):
    """
    Rotates x,y points by the given angle in degrees
    Translation to origin assumed
    """
    rad = angle * pi / 180.0
    xr = xt * cos(rad) - yt * sin(rad)
    yr = yt * cos(rad) + xt * sin(rad)
    return (xr, yr)


class P6Painter(P5Painter):

    def __init__(self):
        P5Painter.__init__(self)
        self.colourRange=(-15.0/255, 15.0/255)
        self.angleRange=(-15.0/360*2*math.pi, 15.0/360*2*math.pi)
        self.radRange=(-3, 2)
        self.base_radius = self.radius

    def get_stroke_endpoints(self, center):
        gradY = center[1] - 1 if (center[1] >= self.intensity_gradient.shape[0]) else center[1]
        gradX = center[0] - 1 if (center[0] >= self.intensity_gradient.shape[1]) else center[0]
        diff = np.array(
            [self.intensity_gradient[gradY, gradX, 1],
             self.intensity_gradient[gradY, gradX, 0]])

        diff = np.array(
            rotate2D(diff[0], diff[1], random.uniform(*self.angleRange)))

        # TODO add a little bit of variance to the diff array

        return ( walk_from(
                    center, self.canvas, self.edgels, 
                    diff, self.halfLen)
               , walk_from(
                    center, self.canvas, self.edgels, 
                    -diff, self.halfLen) )

    def get_colour(self, point):
        colour = P5Painter.get_colour(self, point)
        return (
            max(0, min(1, colour[0] + random.uniform(*self.colourRange))),
            max(0, min(1, colour[1] + random.uniform(*self.colourRange))),
            max(0, min(1, colour[2] + random.uniform(*self.colourRange))))

    def do_paint(self):
        # randomize radius because I did not provide a nice wrapper for rad
        self.radius = max(1,
            self.base_radius + random.randint(*self.radRange))
        P5Painter.do_paint(self)



###########################################################
## Painting helper methods
###########################################################

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

def paintStroke(canvas, p0, p1, colour, rad, alpha):
    # Paint a stroke from pixel p0 = (x0, y0) to pixel p1 = (x1, y1)
    # on the canvas (ny x nx x 3 double array).
    # The stroke has rgb values given by colour (a 3 x 1 vector, with
    # values in [0, 1].  The paintbrush is circular with radius rad>0
    
    sizeIm = canvas.shape[0:2]
    
    mask = markStroke(np.zeros(sizeIm), p0, p1, rad, 1)
    mask = mask * alpha
    
    cmask = np.empty((sizeIm[0], sizeIm[1], 3))
    cmask[:,:,0] = colour[0] * mask
    cmask[:,:,1] = colour[1] * mask
    cmask[:,:,2] = colour[2] * mask
    
    ones = np.empty(mask.shape)
    ones.fill(1.0)
    inverse_mask = ones - mask

    canvas[:,:,0] = canvas[:,:,0] * inverse_mask
    canvas[:,:,1] = canvas[:,:,1] * inverse_mask
    canvas[:,:,2] = canvas[:,:,2] * inverse_mask
    
    canvas[:,:,0:3] = canvas[:,:,0:3] + cmask
    canvas[:,:,3]   = canvas[:,:,3]   + mask

    
#    if random.random() < 0.01:
#        subplot(2,2,1)    
#        imshow(mask, cmap=cm.Greys);
#        subplot(2,2,2)
#        imshow(inverse_mask, cmap=cm.Greys);
#        subplot(2,2,3)    
#        imshow(cmask, cmap=cm.Greys);
#        subplot(2,2,4)
#        imshow(canvas[:,:,0:3])
#        print("showing")
#        show() # This does not block
#        pause(10)
#        sys.exit(1)
    
    return canvas

if __name__ == "__main__":
    # os.chdir("/h/u17/g2/00/g1biggse/code/csc320/a2")
    # os.chdir("/h/u17/g2/00/g4rbage/code/csc320/a2")

    np.set_printoptions(threshold=np.nan)
    import paintrend    
    paintrend.main()