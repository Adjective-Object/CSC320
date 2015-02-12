import sys

DEBUG = True


def debug(*args):
    if DEBUG:
        # using sys.stdout to make it work independent of python version
        sys.stdout.write(" ".join([str(i) for i in args]))
        sys.stdout.write("\n")
        sys.stdout.flush()
