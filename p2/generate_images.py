#!/usr/bin/python

from subprocess import call
import sys
import os

ANACONDA = "/home/mhuan13/anaconda/bin/python"
#ANACONDA = "/local/packages/anaconda3/bin/python"

def paint_and_move_output_to(ddir, image_name, part,
                             radius, length):
    print("part {}, {} -> {}/{}.png".format(
        part, image_name, ddir, image_name))
    
    if not os.path.isdir(ddir):
        os.makedirs(ddir)
    
    call([
        ANACONDA, 
        './paintrend.py',
        '--part={}'.format(part),
        '--image={}'.format(image_name),
        '--out={}/part_{}.png'.format(ddir, part),
        '--radius={}'.format(radius),
        '--length={}'.format(length),
        '--silent'])

if __name__ == '__main__':

    for args in sum([[("orchid", "orchid/small.jpg", i, 3, 3) for i in range(1,7)],
                     [("unsplash_1", "unsplash_1/small.jpg", i, 3, 3) for i in range(7)],
                     [("unsplash_2", "unsplash_2/small.jpg", i, 3, 3) for i in range(7)],
                     [("unsplash_3", "unsplash_3/small.jpg", i, 3, 3) for i in range(7)],
                     [("unsplash_4", "unsplash_4/small.jpg", i, 3, 3) for i in range(7)]
                    ] ,[]):
        paint_and_move_output_to(*args)