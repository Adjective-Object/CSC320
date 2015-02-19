# -*- coding: utf-8 -*-
"""
Created on Sun Feb 15 14:21:03 2015

@author: g4rbage
"""

class Painter(object):
    image = None
    canvas = None

    def give_image(self, image):
        self.image = image
        self.canvas = np.zeros((sizeIm[0], sizeIm[1], 3))
    
    def should_paint(self):
        raise NotImplementedError
    
    def do_paint(self):
        raise NotImplementedError


class P1Painter(Painter):
    def should_paint():
        debug("Amount left unpainted: " + str(np.where(canvas == -1)[0].size))
        return np.where(canvas == -1)[0].size > 0
    
    def do_paint()