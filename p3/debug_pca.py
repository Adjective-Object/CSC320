import itertools, operator, random, math, os, sys

from PIL import Image
from scipy.misc import imread, imshow
import matplotlib.pyplot as plt
import matplotlib

import numpy as np
from pylab import cm

DEBUG = True


# constants for images
IMAGE_WIDTH = 32
IMAGE_HEIGHT = 32
IMAGE_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH)
IMAGE_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT
def unflatten_face(flattened_face):
    ''' reshape a (1024) vector into a (32,32) image
    '''
    return np.reshape(flattened_face, (flattened_face.shape[0] / IMAGE_WIDTH, IMAGE_WIDTH))

def debug(*args):
    if DEBUG:
        # using sys.stdout to make it work independent of python version
        sys.stdout.write(" ".join([str(i) for i in args]))
        sys.stdout.write("\n")
        sys.stdout.flush()

def set_debug(val):
    global DEBUG
    DEBUG = val

def showall(imgs):
    ''' display a list of images as a grid
        (takes either a numpy array of images or a list of numpy arrays that
        are images)
    '''
    font = {'family' : 'normal',
            'size'   : 8}
    matplotlib.rc('font', **font)

    width = math.floor(math.sqrt(imgs.shape[0]))
    height = math.ceil(imgs.shape[0] * 1.0 / width)
    plt.figure().canvas.set_window_title("anaconda")
    for i, img in enumerate(imgs):
        axes = plt.subplot(width, height, i+1)
        axes.get_xaxis().set_visible(False)
        axes.get_yaxis().set_visible(False)
        imgplt = plt.imshow(img, cmap=cm.Greys_r)
        imgplt.set_interpolation('nearest')
        #plt.title(str(i+1))
    plt.show()

def showall_flattened(imgs_flat):
    ''' reshape a seft of (1024) vectors into (32,32) images, and display all of
        them in a greyscale grid
    '''
    print(imgs_flat)
    s=imgs_flat.shape
    unflattened = np.reshape(imgs_flat, (s[0],32,32))
    debug(unflattened.shape)
    showall(unflattened)

def show_flattened_face(flattened_face):
    ''' reshape a (1024) vector into a (32,32) image and display in greyscale
    '''
    unflattened = unflatten_face(flattened_face)
    plt.figure().canvas.set_window_title("anaconda")
    imgplt = plt.imshow(unflattened, cmap=cm.Greys_r)
    imgplt.set_interpolation('nearest')
    plt.show()
