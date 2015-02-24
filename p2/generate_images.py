#!/usr/bin/python

from subprocess import call
import sys
import os

def paint_and_move_output_to(ddir, image_name, part,
                             radius, length):
    print("part {}".format(part))
    
    if not os.path.isdir(ddir):
        os.makedirs(ddir)
    
    call(['/home/mhuan13/anaconda/bin/python', './paintrend.py',
        '--part={}'.format(part),
        '--image={}'.format(image_name),
        '--out={}/part_{}.png'.format(ddir, part),
        '--radius={}'.format(radius),
        '--length={}'.format(length)])

if __name__ == '__main__':
    if len(sys.argv) != 5:
        print('Usage: {} input_image_name output_dir_name radius length alpha'.format(sys.argv[0]))
        sys.exit(1)

    for part in range(2, 7):
        paint_and_move_output_to(sys.argv[2], sys.argv[1], part,
                                 sys.argv[3], sys.argv[4])
