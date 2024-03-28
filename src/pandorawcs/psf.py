# make a class that contains a psf object
import numpy as np
import matplotlib.pyplot as plt
import utils

def psf():
    def __init__(self):
        ...

    def evaluate_xy(self, x, y):
        # returns the psf at the given xy coordinates, centered on (0,0)
        ...

    def evaluate_radial(self, r, th):
         # returns the psf at the given radial coordinates, centered on (0,0)
       ...

    def plot_psf(self):
        ...

def gaussian_psf(psf):
    def __init__(self, std_x, std_y):
