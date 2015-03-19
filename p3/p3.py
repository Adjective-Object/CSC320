#!/local/packages/anaconda3/bin/python
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 10:17:29 2015

@author: g4rbage
"""

import sys
from pca import *

DEBUG = True
def debug(*args):
    if DEBUG:
        # using sys.stdout to make it work independent of python version
        sys.stdout.write(" ".join([str(i) for i in args]))
        sys.stdout.write("\n")
        sys.stdout.flush()

if __name__ == "__main__":
    do_test()
