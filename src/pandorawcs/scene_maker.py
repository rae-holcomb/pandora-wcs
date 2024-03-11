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


    def save_visda(
        self,
        outfile: str = 'pandora_'+Time.now().strftime('%Y-%m-%dT%H:%M:%S')+'_l1_visda.fits',
        rois: bool = False,
        overwrite: bool = True,
    ):
        """
        COPIED FROM BEN'S PANDORASIM, ADJUST TO FIT LATER.
        Function to save FFIs in the FITS format.
        """
        if not hasattr(self, "ffis"):
            raise AttributeError('Please create FFIs first with .get_FFIs() command!')

        corstime = int(np.floor((self.obstime - Time("2000-01-01T12:00:00", scale='utc')).sec))
        finetime = int(corstime % 1 * 10**9 // 1)

        primary_kwds = {
            'EXTNAME': ('PRIMARY', 'name of extension'),
            'NEXTEND': (2, 'number of standard extensions'),
            'SIMDATA': (True, 'simulated data'),
            'SCIDATA': (False, 'science data'),
            'TELESCOP': ('NASA Pandora', 'telescope'),
            'INSTRMNT': ('VISDA', 'instrument'),
            'CREATOR': ('Pandora DPC', 'creator of this product'),
            'CRSOFTV': ('v'+str(__version__), 'creator software version'),
            'TARG_RA': (self.ra.value, 'target right ascension [deg]'),
            'TARG_DEC': (self.dec.value, 'target declination [deg]'),
            'FRMSREQD': (self.ffi_nframes, 'number of frames requested'),
            'FRMSCLCT': (self.ffi_nframes, 'number of frames collected'),
            'NUMCOAD': (1, 'number of frames coadded'),
            'FRMTIME': (self.ffi_nreads * self.VISDA.integration_time.value, 'time in each frame [s]'),
            'EXPDELAY': (-1, 'exposure time delay [ms]'),
            'RICEX': (-1, 'bit noise parameter for Rice compression'),
            'RICEY': (-1, 'bit noise parameter for Rice compression'),
            'CORSTIME': (corstime, 'seconds since the TAI Epoch (12PM Jan 1, 2000)'),
            'FINETIME': (finetime, 'nanoseconds added to CORSTIME seconds'),
        }

        if rois:
            n_arrs, frames, nrows, ncols = self.subarrays.shape

            # Find the next largest perfect square from the number of subarrays given
            next_square = int(np.ceil(np.sqrt(n_arrs)) ** 2)
            sq_sides = int(np.sqrt(next_square))

            # Pad the subarrays with addtional subarrays full of zeros up to the next perfect square
            subarrays = self.subarrays
            padding = np.zeros((next_square - n_arrs, frames, nrows, ncols), dtype=int)
            subarrays = np.append(subarrays, padding, axis=0)

            image_data = (subarrays.reshape(frames, sq_sides, sq_sides, nrows, ncols)
                          .swapaxes(2, 3)
                          .reshape(frames, sq_sides*nrows, sq_sides*ncols))

            roi_data = Table(self.VISDA.corners)

            roitable_kwds = {
                'NAXIS': (2, 'number of array dimensions'),
                'NAXIS1': (len(self.VISDA.corners[0]), 'length of dimension 1'),
                'NAXIS2': (len(self.VISDA.corners), 'length of dimension 2'),
                'PCOUNT': (0, 'number of group parameters'),
                'GCOUNT': (1, 'number of groups'),
                'TFIELDS': (2, 'number of table fields'),
                'TTYPE1': ('Column', 'table field 1 type'),
                'TFORM1': ('I21', 'table field 1 format'),
                'TUNIT1': ('pix', 'table field 1 unit'),
                'TBCOL1': (1, ''),
                'TTYPE2': ('Row', 'table field 2 type'),
                'TFORM2': ('I21', 'table field 2 format'),
                'TUNIT2': ('pix', 'table field 2 unit'),
                'TBCOL2': (22, ''),
                'EXTNAME': ('ROITABLE', 'name of extension'),
                'NROI': (len(self.VISDA.corners), 'number of regions of interest'),
                'ROISTRTX': (-1, 'region of interest origin position in column'),
                'ROISTRTY': (-1, 'region of interest origin position in row'),
                'ROISIZEX': (-1, 'region of interest size in column'),
                'ROISIZEY': (-1, 'region of interest size in row'),
            }
        else:
            image_data = self.ffis

        image_kwds = {
            'NAXIS': (3, 'number of array dimensions'),
            'NAXIS1': (image_data.shape[1], 'first axis size'),  # might need to change these
            'NAXIS2': (image_data.shape[2], 'second axis size'),
            'NAXIS3': (image_data.shape[0], 'third axis size'),
            'EXTNAME': ('SCIENCE', 'extension name'),
            'TTYPE1': ('COUNTS', 'data title: raw pixel counts'),
            'TFORM1': ('J', 'data format: images of unsigned 32-bit integers'),
            'TUNIT1': ('count', 'data units: count'),
        }

        if rois:
            save_to_FITS(
                image_data,
                outfile,
                primary_kwds,
                image_kwds,
                roitable=True,
                roitable_kwds=roitable_kwds,
                roi_data=roi_data,
                overwrite=overwrite)
        else:
            save_to_FITS(image_data, outfile, primary_kwds, image_kwds, overwrite=overwrite)