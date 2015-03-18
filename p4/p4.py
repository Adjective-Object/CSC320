#!/local/packages/anaconda3/bin/python

# ##########################################################################
## Handout painting code.
###########################################################################

import time
import getopt

from scipy.misc import imresize
from PIL import Image
from parts import P1, P2


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
    debug("usage: 'p4.py [options] <back_a> <comp_a> <back_b> <comp_b>")
    debug("     <back_a> is the background of image a")
    debug("     <comp_a> is the composite of the object against backgroud a")
    debug("     <back_b> is the background of image b")
    debug("     <comp_b> is the composite of the object against backgroud b")
    debug("     [options] of")
    debug("         --background <background>")
    debug("             The new background to composite the foreground against")
    debug("             If not specified, this will output an alpha image")
    debug("         --out <fname>")
    debug("             where fname is the name of the output file")
    debug("             defaults to 'out.png'")
    debug("         --silent")
    debug("             suppresses debug messages. defaults to debug messages on")

def parse_opts():
    options, remaining_args = getopt.getopt(
        sys.argv[1:],
        'ohbs:',
        ['out=', 'help=', 'silent=', 'background=']
    )

    out_background = None
    out_name = 'out.png'

    for opt, arg in options:

        elif opt in ('-o', '--out'):
            image_name = arg

        elif opt in ('-b', '--background'):
            radius = float(arg)

        elif opt in ('-h', '--help'):
            print_helptext();
            exit(0);

        elif opt in ('-s', '--silent'):
            global DEBUG
            DEBUG = False
        else:
            debug("unrecognized option/argument pair '%s', '%s'" % (opt, arg))
            debug("%s --help for more info"%(sys.argv[0]));
            sys.exit(1)

    if not painter:
        print_helptext()
        sys.exit(1)

    painter.radius = radius
    painter.halfLen = length / 2
    painter.alpha = alpha

    if len(remaining_args) < 4:
        debug("lacking one of <back_a> <comp_a> <back_b> <comp_b>")
        debug("%s --help for more info"%(sys.argv[0]));
        sys.exit(1)

    return  part, out_background, out_name, remaining_args

def main():
    part, out_background, out_name, imgs = parse_opts();
    
    if out_background:
        part = P2(imgs, out_background)
    else:
        part = P1(imgs)

    out_img = Image(part.execute());
    out_img.save(out_name);

if __name__ == "__main__":
    # os.chdir("/h/u17/g2/00/g1biggse/code/csc320/a2")
    # os.chdir("/h/u17/g2/00/g4rbage/code/csc320/a2")

    np.set_printoptions(threshold=np.nan)
    main()
