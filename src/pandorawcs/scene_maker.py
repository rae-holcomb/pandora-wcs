"""Functions designed to create and manipulate test data."""
import numpy as np
import warnings
import matplotlib.pyplot as plt
# import math
import importlib as imp
import random
import astropy.units as u
from astropy.io import fits
from astropy.wcs import WCS, Sip
from astropy.coordinates import Angle, SkyCoord
from astropy.table import Table
from astroquery.mast import Catalogs

import utils
from utils import SparseWarp3D
import wcs
import detector

import sys
sys.path.insert(0, '../pandora-psf/src')
# import pandorapsf
from scipy import sparse

class SimFile(object):
    """Fill in docstring in design document...
    
    Parameters:
    -----------
    ra: u.Quantity
        Right ascension of the center of the field.

    Returns:
    --------
    catalog : np.ndarray
        X pixel positions in undistorted frame, centered around CRPIX1
    """

    def __init__(self, ra:u.Quantity, dec:u.Quantity, roll:u.Quantity, 
                 detector,  #<- Note: change this later to a generic detector class
                 crpix1: int = None, crpix2: int = None, 
                 affine_M: np.array = utils.make_affine_matrix(),
                 #order: int = 3,             
    ):
        # set class variables
        self.ra = ra
        self.dec = dec
        self.roll = roll
        self.crpix1 = crpix1
        self.crpix2 = crpix2
        self.detector = detector
        self.shape = self.detector.shape

        # set up the basic wcs
        self.wcs = wcs.get_wcs(self.detector, self.ra, self.dec, self.crpix1, self.crpix2, self.roll)

        # get the catalog
        self.cat_true = self._get_gaia_query()

        # check that the affine matrix is the right shape
        if affine_M.shape != (3,3):
            warnings.warn("Affine matrix must have shape (3,3)")
            self.affine_M = utils.make_affine_matrix() # defaults to identity matrix
        else:
            self.affine_M = affine_M  

        # make the warped catalog
        self._apply_affine_transform()

        # make the scene
        # self.scene = self.make_gaussian_scene(catalog='warped')
            
        # bundle into a fits file
        # nor this

    def _get_gaia_query(self) -> Table:
        """
        Queries the gaia catalog in a circle centered on the ra and dec, with a radius large enough to enclose the size of the detector object. Note the object IDs are NOT real. 

        Parameters:
        -----------
        distortion_file: str
            File path to a distortion CSV file.

        Returns:
        --------
        cat : astropy.Table
            Catalog of true source positions with columns ['ra', 'dec', 'gmag', 'gflux', 'source_id', 'pix_x', 'pix_y'].
        """
        ref_coords = SkyCoord(self.ra, self.dec, unit="deg")
        radius = self.detector.naxis1 * (self.detector.pixel_scale) * (np.sqrt(2) * 1.05)  # build in a 5% buffer into the radius we're querying
        gaia_cat = Table(utils.get_sky_catalog(ra=ref_coords.ra.deg, dec=ref_coords.dec.deg, radius=radius))

        # reformat the catalog
        cat = gaia_cat.copy()
        cat['ra'] = gaia_cat['coords'].ra
        cat['dec'] = gaia_cat['coords'].dec
        cat = cat[[ 'ra', 'dec', 'gmag', 'gflux', 'source_id',]]

        # redo the source_id column
        # might change this step later, currently I do this to avoid a column type error when bundling things into a fits table later
        cat['source_id'] = np.arange(len(cat)).astype(int)

        # add in the pixel locations
        pix_x, pix_y = self.wcs.all_world2pix(cat['ra'], cat['dec'], 0)
        cat['pix_x'] = pix_x
        cat['pix_y'] = pix_y

        return cat

    def _apply_affine_transform(self) -> Table:
        """Applies the affine matrix to the true catalog to produce a warped catalog."""
        # make warped catalog
        cat_warp = self.cat_true.copy()
        x1, y1 = utils.apply_affine_transform(self.cat_true['pix_x'].value, self.cat_true['pix_y'].value, crpix1=self.detector.naxis1.value/2, crpix2=self.detector.naxis2.value/2, M=self.affine_M)

        # need to convert from pix back to RA/DEC
        ra1, dec1 = self.wcs.all_pix2world(x1, y1, 0)
        cat_warp['pix_x'] = x1
        cat_warp['pix_y'] = y1
        cat_warp['ra'] = ra1
        cat_warp['dec'] = dec1

        # update the warped catalog to match
        self.cat_warp = cat_warp
        # return cat_warp


    def plot_sources(self) -> None:
        """
        Creates a plot to compare the true and warped catalogs.
        Note for later: What's a good way to show/return figures? Look for pre-existing examples.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=[8,4])
        ax1.scatter(self.cat_true['ra'], self.cat_true['dec'], s=3)
        ax1.scatter(self.cat_warp['ra'], self.cat_warp['dec'], s=3)
        ax1.set_title('Sources by Sky Location')
        ax1.set_xlabel('RA')
        ax1.set_ylabel('DEC')

        ax2.scatter(self.cat_true['pix_x'],self.cat_true['pix_y'], s=3)
        ax2.scatter(self.cat_warp['pix_x'],self.cat_warp['pix_y'], s=3)
        ax2.set_title('Sources by Pixel Location')
        ax2.set_xlabel('naxis1')
        ax2.set_ylabel('naxis2')
        fig.show()

    def make_gaussian_scene(self, catalog='warped', std=(2,2), nstddevs=8) -> np.array:
        """Creates the scene. Use the catalog keyword to specify whether to make the scene from the warped or true catalog.
    
        Parameters:
        -----------
        catalog: str
            If 'warped', will use the warped catalog to make the scene. If 'true', will use the true catalog.
        std: Tuple
            The standard deviation of the gaussian psf in units of pixels in the (x, y) directions.
        nstddevs: int
            The number of standard deviations to calculate the 2D gaussians out to.

        Returns:
        --------
        scene : np.ndarray
            An image populated with sources from the catalog.
        """
        # grab the right catalog 
        if catalog == 'warped':
            cat = self.cat_warp
        elif catalog == 'true':
            cat = self.cat_true
        else:
            warnings.warn("Catalog key word must have value 'warped' or 'true'.")
            return

        mean_x, mean_y = cat['pix_x'], cat['pix_y']
        R, C = np.mgrid[-nstddevs : nstddevs + 1, -nstddevs : nstddevs + 1]

        data = utils.gaussian_2d(
            R[:, :, None],
            C[:, :, None],
            mean_x % 1,
            mean_y % 1,
            np.atleast_1d(std[0]),
            np.atleast_1d(std[1]),
        )

        s = SparseWarp3D(
            data,
            R[:, :, None] + np.floor(mean_x),
            C[:, :, None] + np.floor(mean_y),
            self.shape,
        )

        # Multiply each source by the gmag 
        # NOTE: Might change this later
        w = cat['gmag'].value
        scene = s.dot(w)[0]        
        
        return scene

    def update_gaussian_scene(self, catalog='warped', std=(2,2), nstddevs=8) -> None:
        """Makes a gaussian scene and updates the SimFile object parameters with it."""
        self.scene = self.make_gaussian_scene(catalog=catalog, std=std, nstddevs=nstddevs)

    def plot_scene(self) -> None:
        """Plots the scene.
        Note: need to figure out how best to return figs."""
        fig = plt.imshow(self.scene, vmin=0, vmax=.8)
        return fig

    def to_fits(self) -> fits.hdu.hdulist.HDUList :
        """Converts the current state of the SimFile to a fits file.
    
        Parameters:
        -----------
        warped: str
            If 'warped', will use the warped catalog to make the scene. If 'true', will use the true catalog.

        Returns:
        --------
        scene : np.ndarray
            An image populated with sources from the catalog.
        """
        ...