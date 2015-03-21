#!/local/packages/anaconda3/bin/python
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 10:17:29 2015

@author: g4rbage
"""

import sys, getopt
from pca import *
from debug_pca import debug

def print_helptext():
    debug("usage: '%s [options] validation_dir"%(sys.argv[0]))
    debug("<validation_dir>")
    debug("     subdirectory of <actors_dir>/actor_name the actors")
    debug("     will be read from. defaults to 'validation'")
    debug("[options]")
    debug("    --actors_dir <directory>")
    debug("          folder the actors will be read from.")
    debug("          defaults to 'processed_3'")
    debug("    --silent")
    debug("          suppressing debug messages")
    debug("    --k")
    debug("          comma separated list of k values to check against")
    debug("          defaults to 2,5,10,20,50,80,100,150,200")
    debug("    --mismatches")
    debug("          prints a table showing most common mismatches")

def parse_opts():
    options, remaining_args = getopt.getopt(
        sys.argv[1:],
        'hsma:k:w',
        ['help', 'silent', 'mismatches', 'actors_dir=', 'k=', "write_misses"]
    )

    (debug, p_mismatch, actors_dir) = (True, False, "processed_3")
    write_misses = False
    ks = [2,5,10,20,50,80,100,150,200]

    for opt, arg in options:
        if opt in ('-s', '--silent'):
            debug = False
        elif opt in ('-h', '--help'):
            print_helptext()
            exit(1)
        elif opt in ('-m', '--mismatch'):
            p_mismatch = True
        elif opt in ('-w', '--write_misses'):
            write_misses = True
        elif opt in ('-a', '--actors_dir'):
            actors_dir = arg
        elif opt in ('-k', '--k'):
            ks = [int(a) for a in arg.split(',')]
        else:
            debug("unrecognized option/argument pair '%s', '%s'" %(opt, arg))
            debug("%s --help for more info"%(sys.argv[0]));
            sys.exit(1)

    if len(remaining_args) == 0:
        remaining_args = ["validation"]

    return  debug, write_misses, p_mismatch, actors_dir, remaining_args, ks


if __name__ == "__main__":
    DEBUG, write_misses, p_mismatch, actors_dir, v_dirs, ks = parse_opts()
    set_debug(DEBUG)
    for v_dir in v_dirs:
        do_test(actors_dir=actors_dir,
                display_similarity_table=p_mismatch,
                judge_dir=v_dir,
                k_values=ks,
                SAVE_FACE_MISSES=write_misses)

