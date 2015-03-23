#!/local/packages/anaconda3/bin/python

# ##########################################################################
## Handout painting code.
###########################################################################

import getopt, sys, os

from scipy.misc import imresize, imsave
from PIL import Image
import numpy as np

from parts import *

DEBUG = True
def debug(*args):
    if DEBUG:
        # using sys.stdout to make it work independent of python version
        sys.stdout.write(" ".join([str(i) for i in args]))
        sys.stdout.write("\n")
        sys.stdout.flush()


################################
## Options and main loop logic #
################################


def print_helptext():
    debug("usage: 'p4.py [options] <back_a> <comp_a> <back_b> <comp_b> <part>")
    debug("     <back_a> is the background of image a")
    debug("     <comp_a> is the composite of the object against background a")
    debug("     <back_b> is the background of image b")
    debug("     <comp_b> is the composite of the object against background b")
    debug("     [options] of")
    debug("         --mask")
    debug("             create a mask image instead of the alpha matted image.")
    debug("             incompatible with --background")
    debug("         --background <background>")
    debug("             The new background to composite the foreground against")
    debug("             If not specified, this will output an alpha image")
    debug("         --out <fname>")
    debug("             where fname is the name of the output file")
    debug("             defaults to 'out.png'")
    debug("         --prefix <prefix_fname>")
    debug("             sets back_a comp_a back_b com_b to ")
    debug("             '<prefix_fname>-backA.jpg', <prefix_fname>-compA.jpg, etc")
    debug("         --extension <extension>")
    debug("             changes the file extension used in --prefix")
    debug("             defaults to jpg")
    debug("         --silent")
    debug("             suppresses debug messages. defaults to debug messages on")

def parse_opts():
    options, remaining_args = getopt.getopt(
        sys.argv[1:],
        'o:hb:s:p:e:m',
        ['out=', 'help=', 'silent=', 'background=', 'prefix=', 'extension', 'mask']
    )

    out_background = None
    prefix = None
    mask = False
    suffix = "jpg"
    out_name = 'results/out.png'

    for opt, arg in options:

        if opt in ('-o', '--out'):
            out_name = arg

        elif opt in ('-b', '--background'):
            out_background = arg

        elif opt in ('-p', '--prefix'):
            prefix = arg

        elif opt in ('-e', '--extension'):
            suffix = arg

        elif opt in ('-h', '--help'):
            print_helptext()
            exit(0);

        elif opt in ('-m', '--mask'):
            mask = True

        elif opt in ('-s', '--silent'):
            global DEBUG
            DEBUG = False
        else:
            print("unrecognized option/argument pair '%s', '%s'" % (opt, arg))
            print("%s --help for more info"%(sys.argv[0]))
            sys.exit(1)

    if not prefix and len(remaining_args) < 4:
        print("lacking one of <back_a> <comp_a> <back_b> <comp_b>")
        print("%s --help for more info"%(sys.argv[0]))
        sys.exit(1)

    if mask and out_background:
        print("--mask and --background incompatible args")
        print("%s --help for more info"%(sys.argv[0]))
        sys.exit(1)

    return (mask, out_background, out_name,
            (remaining_args if prefix is None else
                [prefix+"-backA."+suffix,
                 prefix+"-compA."+suffix,
                 prefix+"-backB."+suffix,
                 prefix+"-compB."+suffix]))

def make_folder_for(path):
    """ Traverse path, creating folders if they do not exist """

    traversal_path = path.split("/")[0]
    if traversal_path == "":
        traversal_path = "/"

    for folder in path.split("/")[1:]:
        if not os.path.exists(traversal_path):
            os.mkdir(traversal_path)

        traversal_path = traversal_path + "/" + folder

def main():
    mask, out_background, out_name, image_names = parse_opts()

    debug("parsing into", out_name, "\n\t"+"\n\t".join(image_names) )

    if out_background:
        part = P2(image_names, out_background)
    else:
        part = P1(image_names, mask)

    out_img = part.execute()
    debug("out_image min:", out_img.min(), "out_image max:", out_img.max())

    make_folder_for(out_name)
    imsave(out_name, out_img)

if __name__ == "__main__":
    # os.chdir("/h/u17/g2/00/g1biggse/code/csc320/a2")
    # os.chdir("/h/u17/g2/00/g4rbage/code/csc320/a2")

    np.set_printoptions(threshold=np.nan)
    main()
