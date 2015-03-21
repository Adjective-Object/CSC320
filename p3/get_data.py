from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import random
import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
import os
from scipy.ndimage import filters
import urllib
from PIL import Image


act = ['Aaron Eckhart']#,  'Adam Sandler',   'Adrien Brody',  'Andrea Anders',    'Ashley Benson',    'Christina Applegate',    'Dianna Agron',  'Gillian Anderson']


def timeout(func, args=(), kwargs={}, timeout_duration=1, default=None):
    '''From:
    http://code.activestate.com/recipes/473878-timeout-function-using-threading/'''
    import threading
    class InterruptableThread(threading.Thread):
        def __init__(self):
            threading.Thread.__init__(self)
            self.result = None

        def run(self):
            try:
                self.result = func(*args, **kwargs)
            except:
                self.result = default

    it = InterruptableThread()
    it.start()
    it.join(timeout_duration)
    if it.isAlive():
        return False
    else:
        return it.result

testfile = urllib.URLopener()               

def intensityImg(im):
    
    intensities = im[:,:,0] * 0.30 + im[:,:,1] * 0.59 + im[:,:,2] * 0.11

    #normalize color intensities
    intensities = intensities / np.max(intensities)
    
    return intensities


def processImage(local_file, face_coords, bounds_ratio):
    try:
        img = imread("unprocessed/%s" % (local_file))

        #TODO image_bounds
        real_height = face_coords[3] - face_coords[1]
        new_height = (face_coords[2] - face_coords[0]) * bounds_ratio
        hdiff = int(real_height - new_height)

        img_processed = Image.fromarray(
            img[
                    face_coords[1]:face_coords[3],
                    face_coords[0]+hdiff/2:face_coords[2]-(hdiff-hdiff/2),
                    :]
            ).convert('LA')

        img_thumb = imresize(img_processed, (32, 32))
        print("Saving to: " + local_file)
        img_processed.save("processed/"+local_file,"png")
    except:
        print("error processing %s %s"%(local_file, face_coords))


def doAll():

    bounds_ratio = 0.0
    smallest_width = -1
    for line in open("faces_subset.txt"):
        spl = line.split("\t")
        coords = map(lambda a: int(a), spl[4].split(","))

        width = coords[2] - coords[0]
        c_ratio = float(width) / (coords[3] - coords[1])
        if c_ratio > bounds_ratio:
            bounds_ratio = c_ratio
        if smallest_width == -1 or width < smallest_width:
            smallest_width = width

    print "bounds_ratio: %s, width:%spx"%(bounds_ratio, smallest_width)
    
    for a in act:
        i = 0

        name = a.replace(" ","_")

        if not os.path.exists("unprocessed/"):
            os.mkdir("unprocessed")
        if not os.path.exists("processed/"):
            os.mkdir("processed")

        actor_dirname = "processed/%s" % name
        if not os.path.exists(actor_dirname):
            os.mkdir(actor_dirname)
            os.mkdir(actor_dirname + "/training")
            os.mkdir(actor_dirname + "/test")
            os.mkdir(actor_dirname + "/validation")

        for line in open("faces_subset.txt"):
            if a in line:
                # A version without timeout (uncomment in case you need to 
                # unsupress exceptions, which timeout() does)
                # testfile.retrieve(line.split()[4], "unprocessed/"+filename)
                # timeout is used to stop downloading images which take too long to download

                #  helper variables
                spl = line.split("\t")
                person_name = spl[0].replace(" ","_")
                face_coords = map(lambda a: int(a), spl[4].split(","))
                url = spl[3]
                extension = url.split('.')[-1]
                local_file = "%s/training/%s.%s" % (person_name, i, extension)

                if not os.path.exists("unprocessed/%s"%(person_name)):
                    os.mkdir("unprocessed/%s"%(person_name))

                #load the file with timeout
                timeout(testfile.retrieve, (
                    url, "unprocessed/"+local_file), {}, 5)

                # on fail, print msg and continue
                if not os.path.isfile("unprocessed/"+local_file):
                    print "..fetching file failed <%s>"%(url)
                    continue

                print("processing " + local_file)
                processImage(local_file, face_coords, bounds_ratio)
                i += 1

# print "created processed/%s"%(local_file)
if __name__ == "__main__":
    doAll()