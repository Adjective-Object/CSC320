#!/home/mhuan13/anaconda/bin/python
# TODO change this shebang for the CDF anaconda path
from pylab import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
import sys, glob, os
from math import ceil
from PIL import Image

'''
#######################################
# CONTROL VARIABLES AND DEBUG HELPERS #
#######################################
'''

DEBUG = False
SAVE_CROPPED_IMAGES = False
OUT_PREFIX = "out_"
ALG_NAME="ncc"
FORCE_NO_RESIZE = False
SMALL_IMG_SIZE = 400

def debug_print(*args):
    '''
    Helper function to print something iff the command line argument `--debug`
    has been specified.
    '''
    if DEBUG:
        # using sys.stdout to make it work independent of python version
        sys.stdout.write(" ".join([str(i) for i in args]))
        sys.stdout.write("\n")
        sys.stdout.flush()

def dispImages(*imgs):
    '''
    helper method to show any number of images as a series of images
    in a line on a single figure
    '''
    figure()    
    for i in range(len(imgs)):
        subplot(1,len(imgs),i)
        imshow(imgs[i], cmap=cm.Greys_r)
        title("(" + "x".join((map(str,imgs[i].shape)))+")" )
    show()


'''
################################
# HELPERS FOR SUBIMAGE FINDING #
################################
'''

def group_cuts(cuts, threshold=3):
    '''
    given a list of consecutive integers, finds conscistent runs within 
    <threshold> of each other, and return a tuple of the result.
    
    cuts:
        the list of consecutive integres to group
    
    threshold:
        the maxium difference between adjacent integers in the list before they
        are considered separate groups
    '''
    subgroup = []
    cut_ranges = []
    for cut in cuts:
        if len(subgroup) != 0 and abs(cut - subgroup[-1])>threshold:
            cut_ranges.append((subgroup[0], subgroup[-1]))
            subgroup = []
        subgroup.append(cut)
    cut_ranges.append((subgroup[0], subgroup[-1]))
    return cut_ranges


def get_middle_cuts(cut_ranges, cut_area):
    '''
    given a set of cut ranges, finds the ones most likely to be
    the ones dividing the images (the onees closest to 1/3 and 2/3 of the way
    accross cut_area)

    cut_ranges
        a list of 2-length tuples, specifying the beginning and ending of 
        each cut

    cut_area
        a 2 length tuple specifying the beinning and the end of the area in which
        to look for cuts
    '''
    debug_print("finding the most likely cuts of the set:")
    debug_print("\t",cut_ranges)
    
    avgs = [ mean(c_range) for c_range in cut_ranges]
    left_center = (cut_area[0] + cut_area[1] - cut_area[0]) / 3.0
    right_center = (cut_area[0] + cut_area[1] - cut_area[0])* 2 / 3.0    
    
    closest_l = min(zip(avgs, range(len(avgs))), 
                    key=lambda ar: abs(ar[0] - left_center))[1]
    closest_r = min(zip(avgs, range(len(avgs))), 
                    key=lambda ar: abs(ar[0] - right_center))[1]
    
    debug_print(closest_l, closest_r)    
    
    if closest_l == closest_r:
        av = avgs[closest_l]
        if abs(av - right_center) < abs(av - left_center):
            closest_l += -1
        else:
            closest_r += 1
    
    debug_print(closest_l, closest_r)    
    
    res = [ cut_ranges[i] for i in [closest_l, closest_r] ]    
    debug_print("found %s"%(res))
    
    return res


'''
##################
# BORDER FINDING #
##################
'''


def get_border_and_cuts(img, 
                bordercolor,
                threshold=20,
                maxsize=20, 
                scan_start=0,
                scan_direction='x',
                getcuts=False):
    '''
    scans along a given side of the image, finding the distance into the image
    along that axis that the border is, as well as any "cuts" into the image
    (places where the border extends beyond the specified threshold)

    img
        the image to scan
    
    threshold
        the threshold to pass the image
    
    maxsize
        the maximum size before an image is considered a cut
    
    scan_start
        the position at which the scan should startswith
    
    scan_direction
        the direction of the scan. `x` refers to the left side of the image, 
        `-x` to the right, `y` to the top, and `-y` to the bottom.
    
    getcuts
        if cuts should be calculated and returned.
    '''
    
    hscan = scan_direction.endswith('x')
    if not hscan and not scan_direction.endswith('y'):
        debug_print("invalid scan direction on getborder")
        return 0
    scandir = -1 if scan_direction.startswith('-') else 1
    #debug_print ("scandir: %s"%scandir)

    #expensive scan through edge of image for border
    (height, width) = img.shape
    scans = ['-']*(height if hscan else width)
    for a in range(height if hscan else width):
        scan_spot = (scan_start,a) if hscan else (a,scan_start)
        validZone = False

        for b in range(maxsize if maxsize else (width if hscan else height)):
            if abs(img[scan_spot[1]][scan_spot[0]] - bordercolor) > threshold:
                if validZone:                
                    scans[a] = b
                    break
            else:
                validZone = True
            scan_spot = tuple(
                map(sum,zip(
                    scan_spot, 
                    ((scandir,0) if hscan else (0,scandir))
                )))

    # filtering out cuts from border average calculation
    border = int(ceil(sum(list(filter(lambda x: x!='-', scans))) * (1.0 / len(scans))))

    # finding and grouping infinite borders (cuts through the image)
    cuts = [index for index in range(len(scans)) if scans[index] == '-']
    cut_ranges = group_cuts(cuts)
    if cut_ranges[-1][1] < height:
        cut_ranges.append([height-1, height])
    if cut_ranges[0][0] > 0:
        cut_ranges.append([0, 1])

    middle_cuts = ( get_middle_cuts(cut_ranges, [0,height]) 
                        if getcuts
                        else None)
    return (border, middle_cuts)
    
def find_subimages(in_img):
    ''' 
    finds the information necessary to identify the subimages of a well-formed
    input image.

    in_img
        a well-formed input image, where well-formedness requires the image to]
        have a white external border, a mostly black internal border, and three
        subimages arranged vertically, corresponding to the Blue, Green, and Red
        channels of the composite image, in that order.
    '''
    (height, width) = in_img.shape

    coords = []
    cuts = []
    for (initpos, direction) in [(0,'x'),(0,'y'),(width-1,'-x'),(height-1,'-y')]:
        #debug_print (initpos, direction)
        border_white = get_border_and_cuts(
                            in_img, 255,
                            maxsize=(20*in_img.shape[1]/SMALL_IMG_SIZE),
                            scan_start=initpos,
                            scan_direction=direction)[0]
        
        headstart = 4
        #debug_print ("\tbw=%s"%border_white)
        border_black, blackcuts = get_border_and_cuts(
                            in_img, 17,
                            maxsize=(20*in_img.shape[1]/SMALL_IMG_SIZE),
                            scan_start=(initpos +(border_white + headstart) 
                                * (-1 if direction.startswith('-') else 1)),
                            scan_direction=direction, 
                            getcuts=True)
        border_black += border_white + headstart
        coords.append(border_black)
        
        if direction.endswith('x'):
            cuts += blackcuts

    cuts = get_middle_cuts(cuts, [0, height])
    debug_print("final cuts: %s"%(cuts))
    return coords, cuts

'''
#################
# IMAGE SLICING #
#################
'''

def bounds_to_slice_coords(img, bounds, cuts):
    '''
    converts the image bounds and cuts returned by find_subimage to a series
    of crop coordinates

    img
        the image for which the bounds and cuts are specified

    bounds
        the (left, top, right, bottom) tuple of border sizes

    cuts
        a list of two tuples that divide the three images vertically

    returns
        the crop coordinates of the subimages as a list of tuples
        [.. ((top_left_y, top_left_x), (bottom_right_y, bottom_right_x))..]
    '''
    height, width = img.shape
    if len(cuts) != 2:
        debug_print("fallback - slicing image along coords")
        internalheight = int((height - bounds[1] - bounds[3])/ 3)
        vcut_positions = [(bounds[1] + internalheight * i, 
                          height - bounds[3] + internalheight * (i+1)) 
                              for i in range(3)]
    else:
        vcut_positions = [
                    (bounds[1]      ,cuts[0][0]),
                    (cuts[0][1]     ,cuts[1][0]),  
                    (cuts[1][1]     ,height-bounds[3]),
                    ]
    debug_print(vcut_positions, bounds)

    slice_coords= [(vcut_positions[i][0], 
                    vcut_positions[i][1],                
                    bounds[0],                
                    width - bounds[2]) for i in range(3)]
    return slice_coords

def get_slice_coords(in_img):
    '''
    given an image, find the coordinates at which to crop it to remove the 
    subimages

    in_img
        the image to processing

    returns
        the crop coordinates of the subimages as a list of tuples
        [.. ((top_left_y, top_left_x), (bottom_right_y, bottom_right_x))..]
    '''
    in_img_bounds, in_img_cuts = find_subimages(in_img)
    return bounds_to_slice_coords(in_img, in_img_bounds, in_img_cuts)

def slice_on_coords(img, coords):
    return [img [coords[i][0] : coords[i][1],
                 coords[i][2] : coords[i][3] ] for i in range(3)]


'''
#############################
# HELPERS FOR IMAGE SCORING #
#############################
'''

def get_offset_bounds(base, fit, offset):
    '''
    gets the bounds of the overlapping area between two rectangles
    
    base: 
        a 2-length tuple of the dimensions of the first image (height, width)
    
    fit:
        a 2-length tuple of the dimentsions of the second image (height, width)
        
    offset:
        the bounds of the offset, given in the form  
        (offsety, offsetx)
        with relation to the top left position of the image 
    
    returns
        (offsety, offsetx, height, width)
        a rect inside (fit) to be cropped to fit the overlap
    '''
    
    intersect_rect = (
            int(max(0, offset[0])),
            int(max(0, offset[1])),
            int(min(base[0], offset[0]+fit[0])),
            int(min(base[1], offset[1]+fit[1])))
    #debug_print(intersect_rect)
    return (0 if intersect_rect[0]!=0 else -offset[0],
            0 if intersect_rect[1]!=0 else -offset[1],
            intersect_rect[2]-intersect_rect[0],
            intersect_rect[3]-intersect_rect[1])

def get_overlaps(img_base, img_fit, yoff, xoff):
    '''
    gets the area of overlap between two images, offset by a given position

    img_base:
        the image being fit to

    img_fit
        the image with the offset

    xoff
        the difference on the x axis between top left corner of img_fit 
        and img_base

    yoff
        the difference on the y axis between the top left corner of img_fir
        and img_base

    returns
        a 2-legth tuple of the overlaping sections of either image, with
        img_base first 
    '''
    height_base, width_base = img_base.shape
    height_fit,  width_fit  = img_fit.shape
    
    bounds_fit = get_offset_bounds(
                (height_base, width_base),
                (height_fit,  width_fit),
                (xoff, yoff))
    
    bounds_base = get_offset_bounds(
                (height_fit,  width_fit),
                (height_base, width_base),
                (-xoff, -yoff))

    overlap_a = img_base [
        bounds_base[0]: bounds_base[0]+bounds_base[2],
        bounds_base[1]: bounds_base[1]+bounds_base[3] ]
    
    overlap_b = img_fit [
        bounds_fit[0]: bounds_fit[0]+bounds_fit[2],
        bounds_fit[1]: bounds_fit[1]+bounds_fit[3] ]

    return overlap_a, overlap_b


def get_wander_range(img1, img2, wander=15):
    '''
        given two images, gets the range in which to look for optimal offsets.
        generated by overlapping the centers of the images and offsetting them
        by the wander value

        img1
            the base image

        img2
            the image to be overlaid on img1

        wander
            the distance to look in a given direction
    '''
    (h1, w1) = img1.shape
    (h2, w2) = img2.shape    
    
    init_pos = (int((h1-h2)/2), int((w1-w2)/2))
    return ((init_pos[0]  - wander,
             init_pos[0]  + wander),
             (init_pos[1] - wander,
             init_pos[1]  + wander))

'''
#################
# IMAGE SCORING #
#################
'''


def ssd(img1, img2):
    '''
    given two same-dimension images, returns the sum of squared 
    differences of the images
    '''
    return np.sum((img1 - img2) ** 2 )

def dot(img1, img2):
    '''
    given two same-dimension images, returns the dot product of the images
    '''
    v1, v2 = img1.reshape((img1.size)), img2.reshape((img2.size))
    return np.dot(v1, v2)

def ncc(img1, img2):
    '''
    given two same-dimension images, 
    returns the normalized cross correlation of the images
    '''
    v1, v2 = img1.reshape((img1.size)), img2.reshape((img2.size))
    return np.dot(v1 / np.average(img1), v2 / np.average(img2))

alg_dict = {
    "ncc": ncc,
    "dot": dot,
    "ssd": ssd
}

# scoring algorithms
def score_image(
            img_base,
            img_fit,
            scoring_algorithm,
            max_scan_distance=((-10,10),(-10,10))
            ):
    '''
    attempts to match the R and G channels of an image displayed with all
    channels as subarrays of the greyscale image offset by specified locations
    
    returns:
        the winning result, as a 4 argument tuple, of form (yoff, xoff),
        where `xoff` and `yoff` are the offsets of the top left corner of the 
        images.
    
    img_base:
        2d numpy array of the image to use as a base
    img_fit:
        2d numpy array of the image to use as an overlay
    
    scoring_algorithm:
        a function taking the 2 appropriate array slices and returning 
        some score (higher = better)

    max_scan_distance:
        tuple of the maximum distance to look in a given direction, of format
        ((y_min, y_max), (x_min, x_max))

        max scan distance can be no larger than the minimum of half of either
        the horizontal axis and the vertical access of either image for safe 
        behaviorffi
   
    '''
   
    debug_print("scoring over range %s -> %s"%max_scan_distance )
    
    #score, x_offset, y_offset
    winning_result = (0,None,None)
    for xoff in range (max_scan_distance[1][0], max_scan_distance[1][1]):
        for yoff in range (max_scan_distance[0][0], max_scan_distance[0][1]):
    
                overlap_a, overlap_b = get_overlaps(
                                            img_base, img_fit,
                                            xoff, yoff)
                            
                #debug_print(max_scan_distance, "    " , xoff, yoff)
                
                score = scoring_algorithm(overlap_a, overlap_b)
                
                if(score > winning_result[0]):
                    winning_result = (score, yoff, xoff)
                    #debug_print(winning_result)
                    if winning_result[0] == inf:
                        print "Error in scoring algorithm: ret'd infinite score"
                        exit()

    assert winning_result != (0,None, None)
    return (winning_result[1], winning_result[2])


def get_optimal_offsets(slices, alg=ncc, algname="ncc", 
                                range_g=None, range_r=None):
    '''
    finds the optimal offsets for a series of 3 slices given an algorithms, by
    fitting the second and third images to the first.

    slices:
        3-length tuple of images (blue,red,green) channels

    alg
        the algorithm to use

    algname
        the name of the algorithm to use

    range_g
        the range in which to search for the green image, specified 
        as a tuple of the form ((y_min,x_min), (y_max,x_max))

    range_g
        the range in which to search for the red image, specified 
        as a tuple of the form ((y_min,x_min), (y_max,x_max))

    '''
    debug_print("scoring the offsets")

    r_g = get_wander_range(slices[0], slices[1]) if range_g == None else range_g
    r_r = get_wander_range(slices[0], slices[2]) if range_r == None else range_r
    
    offset_g = score_image(slices[0], slices[1], alg, r_g);
    offset_r = score_image(slices[0], slices[2], alg, r_r);
    
    debug_print ("resulting offsets:\n\t"
            "green-> blue %s, \n\t"
            "red->blue %s"%(offset_g, offset_r))
    
    return offset_g, offset_r


'''
#####################
# IMAGE COMPOSITING #
#####################
'''

# Process an image
def make_composite(slices, offset_g, offset_r):
    '''
    creates a composite by fitting the green and red channels of an image
    onto the blue channel.
    
    slices:
        (b,g,r) tuple of grey images (2d numpy arrays) of each color channel.
        They do not need to be the same size
    
    offset_g:
        (y,x) tuple of the offset of the green channel with relation to the
        blue channel

    offset_r:
        (y,x) tuple of the offset of the red channel with relation to the
        blue channel

    returns:
        an image with all the slices composed into a single image
        
    '''    
    
    margins = ( abs(min(0, offset_g[0], offset_r[0])),
                abs(min(0, offset_g[1], offset_r[1])),
                max(offset_g[0] + slices[1].shape[0], 
                    offset_r[0] + slices[2].shape[0],
                    slices[0].shape[0]),
                max(offset_g[1] + slices[1].shape[1], 
                    offset_r[1] + slices[2].shape[1],
                    slices[0].shape[1]))

    canvas_dim = (margins[0] + margins[2],
                      margins[1] + margins[3])
    
    new_canvas = np.zeros((canvas_dim[0], canvas_dim[1], 3), dtype=uint8)
    
    b_origin = margins[0:2]
    new_canvas [b_origin[0]:slices[0].shape[0] + b_origin[0],
                b_origin[1]:slices[0].shape[1] + b_origin[1],
                2] = slices[0]

    g_origin = list(map(sum, zip(offset_g, margins[0:2])))
    new_canvas [g_origin[0]:slices[1].shape[0] + g_origin[0],
                g_origin[1]:slices[1].shape[1] + g_origin[1],
                1] = slices[1]
    
    r_origin = list(map(sum, zip(offset_r, margins[0:2])))
    new_canvas [r_origin[0]:slices[2].shape[0] + r_origin[0],
                r_origin[1]:slices[2].shape[1] + r_origin[1],
                0] = slices[2]

    return new_canvas

'''
#############################
# HELPERS FOR IMAGE SCALING #
#############################
'''


def conditional_scale(in_img):
    '''
    scales an image if it is too large and the FORCE_NO_RESIZE flag is not set
    '''
    imwidth = in_img.shape[1]*1.0
    debug_print("width: %s"%(imwidth))
    resize_flag = imwidth > SMALL_IMG_SIZE and not FORCE_NO_RESIZE
    resize_factor = SMALL_IMG_SIZE/imwidth
    debug_print("resize flag: %s"%(resize_flag))
    if resize_flag:
        debug_print("resizing image")
        scaled_img = imresize(in_img, resize_factor)
    else:
        scaled_img = in_img
    return scaled_img, resize_flag, resize_factor 

def map_small_to_large(in_img, resize_factor, small_offsets, small_slice_coords, alg, agn):
    '''
    maps the slice coordinates of the 
    '''
    print("mapping downscaled offsets to original image...")
    #scan the range of pixels covered by off sets
    #fix the slices
    range_g = ( (int(math.floor(small_offsets[0][0] - 0.5) / resize_factor), 
                 int(math.ceil( small_offsets[0][0] + 0.5) / resize_factor)),

                (int(math.floor(small_offsets[0][1] - 0.5) / resize_factor), 
                 int(math.ceil( small_offsets[0][1] + 0.5) / resize_factor)))
                
    range_r = ( (int(math.floor(small_offsets[1][0] - 0.5) / resize_factor), 
                 int(math.ceil( small_offsets[1][0] + 0.5) / resize_factor)),

                (int(math.floor(small_offsets[1][1] - 0.5) / resize_factor), 
                 int(math.ceil( small_offsets[1][1] + 0.5) / resize_factor)))

    debug_print("smallmapping: ", small_offsets, resize_factor)
    debug_print("   ", range_g)
    debug_print("   ", range_r)

    slice_coords = [ [int(x/resize_factor) for x in i] for i in small_slice_coords]
    slices = slice_on_coords(in_img, slice_coords)
    offsets = get_optimal_offsets(slices, alg, agn, range_g, range_r)

    return offsets, slices, slice_coords

def as_uint8(in_img):
    '''
    Converts an image of type int32 to a corresponding one of type uint8
    '''
    if in_img.dtype == int32:
        debug_print("image is of type int32, converting to uint8")
        i32 = np.iinfo(np.int32)
        iu8 = np.iinfo(np.uint8)
        in_img = (in_img / 256).astype(uint8, copy=False)
    return in_img


'''
#####################
# MAIN PROGRAM FLOW #
#####################
'''


def processImage(in_img, alg=ncc, agn="ncc"):
    '''
    '''
    in_img = as_uint8(in_img)

    scaled_img, resize_flag, resize_factor = conditional_scale(in_img) 

    debug_print("getting slices of image")
    sys.stdout.flush()
    
    slice_coords = get_slice_coords(scaled_img)
    slices = slice_on_coords(scaled_img, slice_coords)

    debug_print("scoring image")
    sys.stdout.flush()

    offsets = get_optimal_offsets(slices, alg, agn)

    comp = make_composite(slices, offsets[0], offsets[1])

    if resize_flag:
        (offsets, slices, 
            slice_coords) = map_small_to_large(
                                in_img, resize_factor,
                                offsets, slice_coords,
                                alg, agn)

    comp = make_composite(slices, offsets[0], offsets[1])

    return slices, comp, slices, offsets

def save_image(img,prefix,postfix=""):
    '''
    Saves an image to disk with a given file prefix and postfix
    The name of the image will be of form
    <prefix><somenumber><postfix>
    where somenumber begins at 0 and is incremented until it finds
    a file that does not exist.
    '''
    j = Image.fromarray(img)
    fino = 0
    fname = prefix + str(fino) + postfix +".png"
    while(os.path.isfile(fname)):
        fino += 1
        fname = prefix + str(fino) + postfix +".png"
    print ("  saving file %s"%(fname))
    j.save(fname,"png")


def processAll(*image_paths):
    '''
    Processes a series of images, saving the results to disk.
    (see processImage for more details)
    '''
    for path in image_paths:
        print("processing image \"%s\""%(path))
        slices, comp, x, y = processImage(imread(path), alg_dict[ALG_NAME], ALG_NAME)
        if SAVE_CROPPED_IMAGES:
            
            save_image(slices[0],"crop_","_b")
            save_image(slices[1],"crop_","_g")
            save_image(slices[2],"crop_","_r")
            
        save_image(comp,OUT_PREFIX+ALG_NAME+"_")

def main():
    '''
    Reads a series of arguments from sys.argv, setting the appropriate flags,
    and sending the rest of the arguments to processImages
    '''
    global DEBUG, OUT_PREFIX, SAVE_CROPPED_IMAGES, ALG_NAME, FORCE_NO_RESIZE
    args = sys.argv[1:]
    while args[0].startswith("--"):
        if args[0] == "--debug":
            DEBUG = True
            args = args[1:]
        elif args[0] == "--out":
            OUT_PREFIX = args[1]
            args = args[2:]
        elif args[0] == "--cropped":
            SAVE_CROPPED_IMAGES = True
            args = args[1:]
        elif args[0] == "--alg":
            ALG_NAME = args[1]
            args = args[2:]
        elif args[0] == "--noresize":
            FORCE_NO_RESIZE = True
            args = args[1:]
        else:
            print("unrecognized file option %d"%(args[0]))
            exit(1)

    print("alg: %s"%(ALG_NAME))

    if len(args)>=1 and all([not a.startswith("--") for a in args]):
        globs = map(glob.glob, sys.argv[1:])
        files = reduce(lambda a, b: a+b, globs)
        processAll(*files)
    else:
        print("  usage: p1.py [args] [list of file globs]\n")
        print('''

    [--debug]         set the debug flag to true, printing messages verbosely

    [--out prefix]    sets the prefix of the file output. default is 'out_'

    [--cropped]       saves the individual subimages to disk with the prefix 
                      'crop_' and the potfix '_r/g/b' depending on what channel
                      it corresponds to

    [--alg alg_name]  sets the algorithm to match positions with.

    [--noresize]      forces the the algorithm not to downscale large images.
                      (ONLY USE THIS FOR SPEED TESTING, IT IS A TERRIBLE FLAG)
''')


if __name__ == "__main__":
    main()